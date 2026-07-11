package game

import "core:log"
import "core:math/linalg"
import protocol "../protocol"
import transport "../net"

CLIENT_COMMAND_HISTORY :: 64
LOOPBACK_SERVER_PEER :: NetServerPeer(1)

GameNetClientState :: enum {
	Disabled,
	Loopback,
	RemoteConnecting,
	RemoteAccepted,
}

GameNetClient :: struct {
	state:           GameNetClientState,
	host:            transport.Host,
	peer:            transport.Peer,
	map_name:        string,
	content_id:      u32,
	local_player_id: u32,
	command_sequence: u32,
	client_tick:     u32,
	command_accumulator: f32,
	last_server_acked_command: u32,
	last_snapshot_sequence: u32,
	has_snapshot:    bool,
	command_history: [CLIENT_COMMAND_HISTORY]protocol.User_Cmd,
	prediction_history: [CLIENT_COMMAND_HISTORY]PredictedPlayerState,
	last_prediction_error: f32,
	last_prediction_replay_count: u32,
	prediction_correction_count: u32,
	seen_remote_players: [protocol.MAX_SNAPSHOT_PLAYERS + 1]bool,
	receive_buffer:  [protocol.MAX_PACKET_SIZE]byte,
	send_buffer:     [protocol.MAX_PACKET_SIZE]byte,
	local_server_scene: ^Scene,
	local_server:    ^NetServer,
	local_server_accumulator: f32,
}

PredictedPlayerState :: struct {
	sequence:      u32,
	position:      Vec3,
	velocity:      Vec3,
	yaw:           f32,
	pitch:         f32,
	grounded:      bool,
	ground_normal: Vec3,
}

game_net_client_init :: proc(net: ^GameNetClient, config: GameLaunchConfig, scene: ^Scene, assets: ^LoadedSceneAssets) {
	if config.connect_address == "" {
		local_server_scene := new(Scene)
		local_server_scene^ = scene_create(assets, {})
		local_server := new(NetServer)
		local_server^ = net_server_create(local_server_scene, config.map_name, config.content_id)
		net^ = GameNetClient{
			state = .Disabled,
			map_name = config.map_name,
			content_id = config.content_id,
			local_player_id = LOCAL_PLAYER_ID,
			local_server_scene = local_server_scene,
			local_server = local_server,
		}
		if game_net_client_loopback_handshake(net, scene) {
			net.state = .Loopback
		}
		return
	}

	host, ok := transport.client_create()
	if !ok {
		log.warn("Failed to create ENet client")
		return
	}

	peer: transport.Peer
	peer, ok = transport.connect(&host, config.connect_address, config.port)
	if !ok {
		log.warnf("Failed to start connection to %s:%d", config.connect_address, config.port)
		transport.destroy(&host)
		return
	}

	net^ = GameNetClient{
		state = .RemoteConnecting,
		host = host,
		peer = peer,
		map_name = config.map_name,
		content_id = config.content_id,
		local_player_id = LOCAL_PLAYER_ID,
	}
	log.infof("Connecting to server %s:%d map=%s content_id=%08x", config.connect_address, config.port, config.map_name, config.content_id)
}

game_net_client_destroy :: proc(net: ^GameNetClient) {
	if net.host.handle != nil {
		if net.peer != {} {
			transport.disconnect_now(net.peer)
		}
		transport.destroy(&net.host)
	}
	if net.local_server != nil {
		free(net.local_server)
	}
	if net.local_server_scene != nil {
		scene_destroy(net.local_server_scene)
		free(net.local_server_scene)
	}
	net^ = {}
}

game_net_client_reload_local_server_scene :: proc(net: ^GameNetClient, assets: ^LoadedSceneAssets, gpu: SceneGpuResources) {
	if net.local_server_scene == nil {
		return
	}
	scene_reload_assets(net.local_server_scene, assets, gpu)
}

game_net_client_update :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput, look: PlayerLookInput, delta_time: f32) {
	#partial switch net.state {
	case .Disabled:
		// No simulation fallback here: local play should initialize loopback explicitly.
	case .Loopback:
		game_net_client_update_loopback(net, scene, move, look, delta_time)
	case .RemoteConnecting, .RemoteAccepted:
		game_net_client_poll(net, scene)
		if net.state == .RemoteAccepted {
			game_net_client_predict_and_send_user_cmds_for_elapsed_time(net, scene, move, look, delta_time, false)
		}
	}
}

game_net_client_loopback_handshake :: proc(net: ^GameNetClient, scene: ^Scene) -> bool {
	net_server_connect(net.local_server, LOOPBACK_SERVER_PEER)
	packet, ok := protocol.write_client_hello(net.send_buffer[:], net.map_name, net.content_id)
	if !ok {
		log.warn("Failed to write loopback client hello")
		return false
	}
	net_server_handle_packet(net.local_server, LOOPBACK_SERVER_PEER, protocol.CHANNEL_CONTROL, packet)
	return game_net_client_drain_loopback_server(net, scene)
}

game_net_client_update_loopback :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput, look: PlayerLookInput, delta_time: f32) {
	cmds := game_net_client_predict_and_queue_user_cmds(net, scene, move, look, delta_time, true)
	if cmds.count > 0 {
		packet_cmds := game_net_client_recent_user_cmds(net)
		packet, ok := protocol.write_user_cmds(net.send_buffer[:], packet_cmds)
		if !ok {
			log.warnf("Failed to write loopback user cmds newest_sequence=%d count=%d", net.command_sequence, packet_cmds.count)
			return
		}
		net_server_handle_packet(net.local_server, LOOPBACK_SERVER_PEER, protocol.CHANNEL_USER_CMDS, packet)
	}
	net.local_server_accumulator += min(delta_time, 0.25)
	for net.local_server_accumulator >= NET_SERVER_TICK_TIME {
		net_server_tick(net.local_server)
		net.local_server_accumulator -= NET_SERVER_TICK_TIME
	}
	game_net_client_drain_loopback_server(net, scene)
}

game_net_client_poll :: proc(net: ^GameNetClient, scene: ^Scene) {
	for _ in 0..<8 {
		event := transport.poll(&net.host, net.receive_buffer[:], 0)
		switch event.kind {
		case .None:
			return
		case .Connect:
			log.infof("Connected to server peer=%v", event.peer)
			packet, ok := protocol.write_client_hello(net.send_buffer[:], net.map_name, net.content_id)
			if !ok || !transport.send(net.peer, protocol.CHANNEL_CONTROL, packet, .Reliable) {
				log.warn("Failed to send client hello")
				continue
			}
			transport.flush(&net.host)
		case .Disconnect:
			log.warn("Disconnected from server")
			game_net_client_destroy(net)
			return
		case .Receive:
			game_net_client_handle_packet(net, scene, event)
		}
	}
}

game_net_client_handle_packet :: proc(net: ^GameNetClient, scene: ^Scene, event: transport.Event) {
	if event.truncated {
		log.warnf("Dropped truncated server packet channel=%d bytes=%d", event.channel, len(event.data))
		return
	}

	game_net_client_handle_server_packet(net, scene, event.data, event.channel)
}

game_net_client_handle_server_packet :: proc(net: ^GameNetClient, scene: ^Scene, data: []byte, channel: u8) {
	packet, err := protocol.parse_packet(data, channel)
	if err != .None {
		log.warnf("Rejected server packet channel=%d bytes=%d err=%v", channel, len(data), err)
		return
	}

	#partial switch packet.header.kind {
	case .Server_Hello:
		hello := packet.hello
		hello_map := hello.map_name
		if !protocol.map_name_equals_string(&hello_map, net.map_name) || hello.content_id != net.content_id {
			log.warnf("Server hello mismatch map=%s content_id=%08x", protocol.map_name_string(&hello_map), hello.content_id)
			return
		}
		net.local_player_id = hello.player_id
		scene.camera_player_id = hello.player_id
		scene_reset_player_to_spawn(scene, hello.player_id)
		if net.local_server != nil {
			net.state = .Loopback
			log.infof("Started local loopback client/server session map=%s content_id=%08x player_id=%d", protocol.map_name_string(&hello_map), hello.content_id, hello.player_id)
		} else {
			net.state = .RemoteAccepted
			log.infof("Server accepted network session map=%s content_id=%08x player_id=%d", protocol.map_name_string(&hello_map), hello.content_id, hello.player_id)
		}
	case .Server_Reject:
		log.warnf("Server rejected network session reason=%v", packet.reject_reason)
		game_net_client_destroy(net)
	case .Server_Snapshot:
		game_net_client_apply_snapshot(net, scene, packet.snapshot)
	case:
		log.warnf("Unexpected server packet kind=%v", packet.header.kind)
	}
}

game_net_client_drain_loopback_server :: proc(net: ^GameNetClient, scene: ^Scene) -> bool {
	accepted := net.state == .Loopback
	for {
		packet, ok := net_server_poll_outgoing(net.local_server)
		if !ok {
			break
		}
		if packet.peer != LOOPBACK_SERVER_PEER {
			continue
		}
		game_net_client_handle_server_packet(net, scene, packet.data[:packet.length], packet.channel)
		accepted = accepted || net.state == .Loopback
	}
	return accepted
}

game_net_client_predict_and_send_user_cmds_for_elapsed_time :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput, look: PlayerLookInput, delta_time: f32, step_world: bool) {
	cmds := game_net_client_predict_and_queue_user_cmds(net, scene, move, look, delta_time, step_world)
	if cmds.count > 0 {
		packet_cmds := game_net_client_recent_user_cmds(net)
		packet, ok := protocol.write_user_cmds(net.send_buffer[:], packet_cmds)
		if !ok || !transport.send(net.peer, protocol.CHANNEL_USER_CMDS, packet, .Unreliable) {
			log.warnf("Failed to send user cmds newest_sequence=%d count=%d", net.command_sequence, packet_cmds.count)
			return
		}
		transport.flush(&net.host)
	}
}

game_net_client_predict_and_queue_user_cmds :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput, look: PlayerLookInput, delta_time: f32, step_world: bool) -> protocol.User_Cmds {
	player := scene_player(scene, net.local_player_id)
	if player == nil {
		return {}
	}
	player_apply_look(player, look)

	queued: protocol.User_Cmds
	net.command_accumulator += min(delta_time, 0.25)
	for net.command_accumulator >= NET_SERVER_TICK_TIME {
		cmd := game_net_client_next_user_cmd(net, move, player.yaw, player.pitch)
		game_net_client_store_user_cmd(net, cmd)
		game_net_client_predict_user_cmd(net, scene, cmd, step_world)
		if queued.count < protocol.MAX_USER_CMDS_PER_PACKET {
			queued.cmds[int(queued.count)] = cmd
			queued.count += 1
		}
		net.command_accumulator -= NET_SERVER_TICK_TIME
	}
	return queued
}

game_net_client_predict_user_cmd :: proc(net: ^GameNetClient, scene: ^Scene, cmd: protocol.User_Cmd, step_world: bool) -> bool {
	player := scene_player(scene, net.local_player_id)
	if player == nil {
		return false
	}
	player.yaw = cmd.yaw
	player.pitch = cmd.pitch
	move := player_move_input_from_user_cmd(cmd)
	result := player_update(player, &scene.physics, move, NET_SERVER_TICK_TIME)
	scene_touch_player(scene, player, result)
	if step_world {
		scene_step_physics(scene, NET_SERVER_TICK_TIME)
	}
	game_net_client_store_predicted_state(net, cmd.sequence, player^)
	return true
}

game_net_client_apply_snapshot :: proc(net: ^GameNetClient, scene: ^Scene, snapshot: protocol.Server_Snapshot) {
	if net.has_snapshot && snapshot.sequence <= net.last_snapshot_sequence {
		return
	}
	net.has_snapshot = true
	net.last_snapshot_sequence = snapshot.sequence
	if snapshot.server_tick > REMOTE_INTERPOLATION_DELAY_TICKS {
		scene.remote_render_tick = snapshot.server_tick - REMOTE_INTERPOLATION_DELAY_TICKS
	} else {
		scene.remote_render_tick = 0
	}

	if snapshot.last_processed_user_cmd > net.last_server_acked_command {
		net.last_server_acked_command = snapshot.last_processed_user_cmd
	}
	for i in 0..<int(snapshot.player_count) {
		state := snapshot.players[i]
		if state.player_id == net.local_player_id {
			game_net_client_reconcile_local_player(net, scene, state, snapshot.last_processed_user_cmd)
			continue
		}
		if state.player_id < len(net.seen_remote_players) && !net.seen_remote_players[state.player_id] {
			net.seen_remote_players[state.player_id] = true
			log.infof("Remote player visible local=%d remote=%d", net.local_player_id, state.player_id)
		}
		scene_upsert_remote_player_sample(
			scene,
			state.player_id,
			{state.position.x, state.position.y, state.position.z},
			state.yaw,
			snapshot.server_tick,
		)
	}
	for i in 0..<int(snapshot.removed_prop_count) {
		scene_remove_replicated_prop(scene, snapshot.removed_prop_ids[i])
	}
	for i in 0..<int(snapshot.prop_count) {
		state := snapshot.props[i]
		rotation: linalg.Quaternionf32
		rotation.x = state.rotation.x
		rotation.y = state.rotation.y
		rotation.z = state.rotation.z
		rotation.w = state.rotation.w
		scene_upsert_replicated_prop(
			scene,
			state.net_id,
			state.prop_asset_index,
			{state.position.x, state.position.y, state.position.z},
			rotation,
			snapshot.server_tick,
		)
	}
}

game_net_client_reconcile_local_player :: proc(net: ^GameNetClient, scene: ^Scene, state: protocol.Server_Player_State, ack_sequence: u32) {
	authoritative := predicted_state_from_server_player_state(ack_sequence, state)
	if predicted, ok := game_net_client_predicted_state(net, ack_sequence); ok {
		delta := authoritative.position - predicted.position
		net.last_prediction_error = linalg.length(delta)
		if net.last_prediction_error > 0.001 {
			net.prediction_correction_count += 1
		}
	} else {
		net.last_prediction_error = 0
	}

	scene_upsert_local_authoritative_player(scene, state)
	net.last_prediction_replay_count = 0
	for sequence := ack_sequence + 1; sequence <= net.command_sequence; sequence += 1 {
		cmd := net.command_history[int(sequence % CLIENT_COMMAND_HISTORY)]
		if cmd.sequence != sequence {
			continue
		}
		if game_net_client_predict_user_cmd(net, scene, cmd, false) {
			net.last_prediction_replay_count += 1
		}
	}
}

scene_upsert_local_authoritative_player :: proc(scene: ^Scene, state: protocol.Server_Player_State) {
	player := scene_ensure_player(scene, state.player_id, {state.position.x, state.position.y, state.position.z}, state.yaw)
	player.position = {state.position.x, state.position.y, state.position.z}
	player.velocity = {state.velocity.x, state.velocity.y, state.velocity.z}
	player.yaw = state.yaw
	player.pitch = state.pitch
	player.grounded = state.grounded
	player.ground_normal = {state.ground_normal.x, state.ground_normal.y, state.ground_normal.z}
}

game_net_client_store_user_cmd :: proc(net: ^GameNetClient, cmd: protocol.User_Cmd) {
	net.command_history[int(cmd.sequence % CLIENT_COMMAND_HISTORY)] = cmd
}

game_net_client_store_predicted_state :: proc(net: ^GameNetClient, sequence: u32, player: PlayerController) {
	net.prediction_history[int(sequence % CLIENT_COMMAND_HISTORY)] = predicted_state_from_player(sequence, player)
}

game_net_client_predicted_state :: proc(net: ^GameNetClient, sequence: u32) -> (PredictedPlayerState, bool) {
	if sequence == 0 {
		return {}, false
	}
	state := net.prediction_history[int(sequence % CLIENT_COMMAND_HISTORY)]
	return state, state.sequence == sequence
}

predicted_state_from_player :: proc(sequence: u32, player: PlayerController) -> PredictedPlayerState {
	return {
		sequence = sequence,
		position = player.position,
		velocity = player.velocity,
		yaw = player.yaw,
		pitch = player.pitch,
		grounded = player.grounded,
		ground_normal = player.ground_normal,
	}
}

predicted_state_from_server_player_state :: proc(sequence: u32, state: protocol.Server_Player_State) -> PredictedPlayerState {
	return {
		sequence = sequence,
		position = {state.position.x, state.position.y, state.position.z},
		velocity = {state.velocity.x, state.velocity.y, state.velocity.z},
		yaw = state.yaw,
		pitch = state.pitch,
		grounded = state.grounded,
		ground_normal = {state.ground_normal.x, state.ground_normal.y, state.ground_normal.z},
	}
}

game_net_client_recent_user_cmds :: proc(net: ^GameNetClient) -> protocol.User_Cmds {
	cmds: protocol.User_Cmds
	if net.command_sequence == 0 {
		return cmds
	}

	first_sequence := net.last_server_acked_command + 1
	oldest_available := u32(1)
	if net.command_sequence > CLIENT_COMMAND_HISTORY - 1 {
		oldest_available = net.command_sequence - (CLIENT_COMMAND_HISTORY - 1)
	}
	if first_sequence < oldest_available {
		first_sequence = oldest_available
	}
	max_count := u32(protocol.MAX_USER_CMDS_PER_PACKET)
	if net.command_sequence - first_sequence + 1 > max_count {
		first_sequence = net.command_sequence - max_count + 1
	}

	for sequence := first_sequence; sequence <= net.command_sequence; sequence += 1 {
		cmd := net.command_history[int(sequence % CLIENT_COMMAND_HISTORY)]
		if cmd.sequence != sequence {
			continue
		}
		cmds.cmds[int(cmds.count)] = cmd
		cmds.count += 1
	}
	return cmds
}

game_net_client_next_user_cmd :: proc(net: ^GameNetClient, move: PlayerMoveInput, yaw, pitch: f32) -> protocol.User_Cmd {
	net.command_sequence += 1
	net.client_tick += 1
	return game_net_client_make_user_cmd(net, move, yaw, pitch)
}

game_net_client_make_user_cmd :: proc(net: ^GameNetClient, move: PlayerMoveInput, yaw, pitch: f32) -> protocol.User_Cmd {
	buttons := u16(0)
	if move.jump_held {
		buttons |= protocol.BUTTON_JUMP
	}

	return protocol.User_Cmd{
		sequence = net.command_sequence,
		client_tick = net.client_tick,
		buttons = buttons,
		move_forward = move.move_forward,
		move_right = move.move_right,
		yaw = yaw,
		pitch = pitch,
	}
}

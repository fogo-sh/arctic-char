package game

import "core:log"
import protocol "../protocol"
import transport "../net"

LOOPBACK_PACKET_CAPACITY :: 16
CLIENT_COMMAND_HISTORY :: 64

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
	last_server_acked_command: u32,
	last_snapshot_sequence: u32,
	has_snapshot:    bool,
	command_history: [CLIENT_COMMAND_HISTORY]protocol.User_Cmd,
	seen_remote_players: [protocol.MAX_SNAPSHOT_PLAYERS + 1]bool,
	receive_buffer:  [protocol.MAX_PACKET_SIZE]byte,
	send_buffer:     [protocol.MAX_PACKET_SIZE]byte,
	client_to_server: GameLoopbackQueue,
	server_to_client: GameLoopbackQueue,
}

GameLoopbackPacket :: struct {
	channel: u8,
	length:  int,
	data:    [protocol.MAX_PACKET_SIZE]byte,
}

GameLoopbackQueue :: struct {
	packets: [LOOPBACK_PACKET_CAPACITY]GameLoopbackPacket,
	get:     u32,
	send:    u32,
}

game_net_client_init :: proc(net: ^GameNetClient, config: GameLaunchConfig) {
	if config.connect_address == "" {
		net^ = GameNetClient{
			state = .Disabled,
			map_name = config.map_name,
			content_id = config.content_id,
			local_player_id = LOCAL_PLAYER_ID,
		}
		if game_net_client_loopback_handshake(net) {
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
	net^ = {}
}

game_net_client_update :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput, look: PlayerLookInput, delta_time: f32) {
	#partial switch net.state {
	case .Disabled:
		// No simulation fallback here: local play should initialize loopback explicitly.
	case .Loopback:
		game_net_client_send_loopback_user_cmd(net, scene, move, look, delta_time)
	case .RemoteConnecting, .RemoteAccepted:
		// Temporary prediction path until the dedicated server shares the same scene simulation.
		scene_update(scene, net.local_player_id, move, look, delta_time)
		game_net_client_poll(net, scene)
		if net.state == .RemoteAccepted {
			game_net_client_send_user_cmd(net, scene, move)
		}
	}
}

game_net_client_loopback_handshake :: proc(net: ^GameNetClient) -> bool {
	packet, ok := protocol.write_client_hello(net.send_buffer[:], net.map_name, net.content_id)
	if !ok {
		log.warn("Failed to write loopback client hello")
		return false
	}
	if !game_loopback_send(&net.client_to_server, protocol.CHANNEL_CONTROL, packet) {
		return false
	}

	client_packet, packet_ok := game_loopback_poll(&net.client_to_server)
	if !packet_ok {
		log.warn("Loopback client hello was not queued")
		return false
	}
	parsed, err := protocol.parse_packet(client_packet.data[:client_packet.length], client_packet.channel)
	if err != .None || parsed.header.kind != .Client_Hello {
		log.warnf("Loopback client hello parse failed err=%v", err)
		return false
	}

	response: []byte
	response, ok = protocol.write_server_hello(net.receive_buffer[:], net.map_name, net.content_id, LOCAL_PLAYER_ID)
	if !ok {
		log.warn("Failed to write loopback server hello")
		return false
	}
	if !game_loopback_send(&net.server_to_client, protocol.CHANNEL_CONTROL, response) {
		return false
	}
	queued_response: GameLoopbackPacket
	queued_response, packet_ok = game_loopback_poll(&net.server_to_client)
	if !packet_ok {
		log.warn("Loopback server hello was not queued")
		return false
	}
	server_packet: protocol.Parsed_Packet
	server_packet, err = protocol.parse_packet(queued_response.data[:queued_response.length], queued_response.channel)
	if err != .None || server_packet.header.kind != .Server_Hello {
		log.warnf("Loopback server hello parse failed err=%v", err)
		return false
	}
	net.local_player_id = server_packet.hello.player_id
	log.infof("Started local loopback client/server session map=%s content_id=%08x", net.map_name, net.content_id)
	return true
}

game_net_client_send_loopback_user_cmd :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput, look: PlayerLookInput, delta_time: f32) {
	net.command_sequence += 1
	net.client_tick += 1
	player := scene_player(scene, net.local_player_id)
	if player == nil {
		return
	}
	yaw, pitch := player_look_angles_after_input(player^, look)
	cmd := game_net_client_make_user_cmd(net, move, yaw, pitch)
	game_net_client_store_user_cmd(net, cmd)
	packet, ok := protocol.write_user_cmd(net.send_buffer[:], cmd)
	if !ok {
		log.warnf("Failed to write loopback user cmd sequence=%d", cmd.sequence)
		return
	}
	if !game_loopback_send(&net.client_to_server, protocol.CHANNEL_USER_CMDS, packet) {
		return
	}
	queued_cmd, packet_ok := game_loopback_poll(&net.client_to_server)
	if !packet_ok {
		log.warnf("Loopback user cmd was not queued sequence=%d", cmd.sequence)
		return
	}
	parsed, err := protocol.parse_packet(queued_cmd.data[:queued_cmd.length], queued_cmd.channel)
	if err != .None || parsed.header.kind != .User_Cmd {
		log.warnf("Loopback user cmd parse failed sequence=%d err=%v", cmd.sequence, err)
		return
	}
	if parsed.user_cmds.count == 0 {
		return
	}
	scene_update_from_user_cmd(scene, net.local_player_id, parsed.user_cmds.cmds[int(parsed.user_cmds.count) - 1], delta_time)
}

game_loopback_send :: proc(queue: ^GameLoopbackQueue, channel: u8, data: []byte) -> bool {
	if len(data) > protocol.MAX_PACKET_SIZE {
		log.warnf("Dropped oversized loopback packet channel=%d bytes=%d", channel, len(data))
		return false
	}
	if queue.send - queue.get >= LOOPBACK_PACKET_CAPACITY {
		if channel == protocol.CHANNEL_CONTROL {
			log.warn("Dropped loopback control packet: queue is full")
			return false
		}
		log.warnf("Dropping oldest loopback packet: queue is full channel=%d", channel)
		queue.get += 1
	}
	index := queue.send % LOOPBACK_PACKET_CAPACITY
	packet := &queue.packets[index]
	packet.channel = channel
	packet.length = len(data)
	copy(packet.data[:], data[:packet.length])
	queue.send += 1
	return true
}

game_loopback_poll :: proc(queue: ^GameLoopbackQueue) -> (packet: GameLoopbackPacket, ok: bool) {
	if queue.send - queue.get > LOOPBACK_PACKET_CAPACITY {
		queue.get = queue.send - LOOPBACK_PACKET_CAPACITY
	}
	if queue.get >= queue.send {
		return {}, false
	}
	index := queue.get % LOOPBACK_PACKET_CAPACITY
	queue.get += 1
	return queue.packets[index], true
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

	packet, err := protocol.parse_packet(event.data, event.channel)
	if err != .None {
		log.warnf("Rejected server packet channel=%d bytes=%d err=%v", event.channel, len(event.data), err)
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
		net.state = .RemoteAccepted
		log.infof("Server accepted network session map=%s content_id=%08x player_id=%d", protocol.map_name_string(&hello_map), hello.content_id, hello.player_id)
	case .Server_Reject:
		log.warnf("Server rejected network session reason=%v", packet.reject_reason)
		game_net_client_destroy(net)
	case .Server_Snapshot:
		game_net_client_apply_snapshot(net, scene, packet.snapshot)
	case:
		log.warnf("Unexpected server packet kind=%v", packet.header.kind)
	}
}

game_net_client_send_user_cmd :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput) {
	net.command_sequence += 1
	net.client_tick += 1
	player := scene_player(scene, net.local_player_id)
	if player == nil {
		return
	}
	cmd := game_net_client_make_user_cmd(net, move, player.yaw, player.pitch)
	game_net_client_store_user_cmd(net, cmd)
	cmds := game_net_client_recent_user_cmds(net)
	packet, ok := protocol.write_user_cmds(net.send_buffer[:], cmds)
	if !ok || !transport.send(net.peer, protocol.CHANNEL_USER_CMDS, packet, .Unreliable) {
		log.warnf("Failed to send user cmds newest_sequence=%d count=%d", cmd.sequence, cmds.count)
		return
	}
	transport.flush(&net.host)
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
}

game_net_client_store_user_cmd :: proc(net: ^GameNetClient, cmd: protocol.User_Cmd) {
	net.command_history[int(cmd.sequence % CLIENT_COMMAND_HISTORY)] = cmd
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

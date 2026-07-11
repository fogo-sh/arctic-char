package game

import "core:log"
import "core:math"
import "core:math/linalg"
import protocol "../protocol"
import transport "../net"

LOOPBACK_PACKET_CAPACITY :: 16

GameNetClientState :: enum {
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
	command_sequence: u32,
	client_tick:     u32,
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
			state = .Loopback,
			map_name = config.map_name,
			content_id = config.content_id,
		}
		game_net_client_loopback_handshake(net)
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
	case .Loopback:
		game_net_client_send_loopback_user_cmd(net, scene, move, look, delta_time)
	case .RemoteConnecting, .RemoteAccepted:
		// Temporary prediction path until the dedicated server shares the same scene simulation.
		scene_update(scene, move, look, delta_time)
		game_net_client_poll(net)
		if net.state == .RemoteAccepted {
			game_net_client_send_user_cmd(net, scene, move)
		}
	}
}

game_net_client_loopback_handshake :: proc(net: ^GameNetClient) {
	packet, ok := protocol.write_client_hello(net.send_buffer[:], net.map_name, net.content_id)
	if !ok {
		log.warn("Failed to write loopback client hello")
		return
	}
	game_loopback_send(&net.client_to_server, protocol.CHANNEL_CONTROL, packet)

	client_packet, packet_ok := game_loopback_poll(&net.client_to_server)
	if !packet_ok {
		log.warn("Loopback client hello was not queued")
		return
	}
	parsed, err := protocol.parse_packet(client_packet.data[:client_packet.length], client_packet.channel)
	if err != .None || parsed.header.kind != .Client_Hello {
		log.warnf("Loopback client hello parse failed err=%v", err)
		return
	}

	response: []byte
	response, ok = protocol.write_server_hello(net.receive_buffer[:], net.map_name, net.content_id)
	if !ok {
		log.warn("Failed to write loopback server hello")
		return
	}
	game_loopback_send(&net.server_to_client, protocol.CHANNEL_CONTROL, response)
	queued_response: GameLoopbackPacket
	queued_response, packet_ok = game_loopback_poll(&net.server_to_client)
	if !packet_ok {
		log.warn("Loopback server hello was not queued")
		return
	}
	server_packet: protocol.Parsed_Packet
	server_packet, err = protocol.parse_packet(queued_response.data[:queued_response.length], queued_response.channel)
	if err != .None || server_packet.header.kind != .Server_Hello {
		log.warnf("Loopback server hello parse failed err=%v", err)
		return
	}
	log.infof("Started local loopback client/server session map=%s content_id=%08x", net.map_name, net.content_id)
}

game_net_client_send_loopback_user_cmd :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput, look: PlayerLookInput, delta_time: f32) {
	net.command_sequence += 1
	net.client_tick += 1
	yaw := scene.player.yaw + look.look_delta.x * PLAYER_SPEC.mouse_sensitivity
	pitch := scene.player.pitch - look.look_delta.y * PLAYER_SPEC.mouse_sensitivity
	pitch = math.clamp(pitch, linalg.to_radians(f32(-89)), linalg.to_radians(f32(89)))
	cmd := game_net_client_make_user_cmd(net, move, yaw, pitch)
	packet, ok := protocol.write_user_cmd(net.send_buffer[:], cmd)
	if !ok {
		log.warnf("Failed to write loopback user cmd sequence=%d", cmd.sequence)
		return
	}
	game_loopback_send(&net.client_to_server, protocol.CHANNEL_USER_CMDS, packet)
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
	scene_update_from_user_cmd(scene, parsed.user_cmd, delta_time)
}

game_loopback_send :: proc(queue: ^GameLoopbackQueue, channel: u8, data: []byte) {
	index := queue.send % LOOPBACK_PACKET_CAPACITY
	packet := &queue.packets[index]
	packet.channel = channel
	packet.length = min(len(data), protocol.MAX_PACKET_SIZE)
	copy(packet.data[:], data[:packet.length])
	queue.send += 1
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

game_net_client_poll :: proc(net: ^GameNetClient) {
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
			game_net_client_handle_packet(net, event)
		}
	}
}

game_net_client_handle_packet :: proc(net: ^GameNetClient, event: transport.Event) {
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
		net.state = .RemoteAccepted
		log.infof("Server accepted network session map=%s content_id=%08x", protocol.map_name_string(&hello_map), hello.content_id)
	case .Server_Reject:
		log.warnf("Server rejected network session reason=%v", packet.reject_reason)
		game_net_client_destroy(net)
	case .Server_Player_State:
		state := packet.player_state
		scene_upsert_remote_player(
			&g.scene,
			state.player_id,
			{state.position.x, state.position.y, state.position.z},
			state.yaw,
		)
	case:
		log.warnf("Unexpected server packet kind=%v", packet.header.kind)
	}
}

game_net_client_send_user_cmd :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput) {
	net.command_sequence += 1
	net.client_tick += 1
	cmd := game_net_client_make_user_cmd(net, move, scene.player.yaw, scene.player.pitch)
	packet, ok := protocol.write_user_cmd(net.send_buffer[:], cmd)
	if !ok || !transport.send(net.peer, protocol.CHANNEL_USER_CMDS, packet, .Unreliable) {
		log.warnf("Failed to send user cmd sequence=%d", cmd.sequence)
		return
	}
	transport.flush(&net.host)
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

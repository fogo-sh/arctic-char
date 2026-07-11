package game

import "core:log"
import protocol "../protocol"
import transport "../net"

GameNetClientState :: enum {
	Offline,
	Connecting,
	Accepted,
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
}

game_net_client_init :: proc(net: ^GameNetClient, config: GameLaunchConfig) {
	if config.connect_address == "" {
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
		state = .Connecting,
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

game_net_client_update :: proc(net: ^GameNetClient, scene: ^Scene, move: PlayerMoveInput) {
	if net.state == .Offline {
		return
	}

	game_net_client_poll(net)
	if net.state == .Accepted {
		game_net_client_send_user_cmd(net, scene, move)
	}
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
		net.state = .Accepted
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
	buttons := u16(0)
	if move.jump_held {
		buttons |= protocol.BUTTON_JUMP
	}

	cmd := protocol.User_Cmd{
		sequence = net.command_sequence,
		client_tick = net.client_tick,
		buttons = buttons,
		move_forward = move.move_forward,
		move_right = move.move_right,
		yaw = scene.player.yaw,
		pitch = scene.player.pitch,
	}
	packet, ok := protocol.write_user_cmd(net.send_buffer[:], cmd)
	if !ok || !transport.send(net.peer, protocol.CHANNEL_USER_CMDS, packet, .Unreliable) {
		log.warnf("Failed to send user cmd sequence=%d", cmd.sequence)
		return
	}
	transport.flush(&net.host)
}

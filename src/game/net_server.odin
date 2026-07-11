package game

import "core:log"
import protocol "../protocol"
import transport "../net"

NET_SERVER_MAX_CLIENTS :: 32
NET_SERVER_PENDING_USER_CMDS :: 64
NET_SERVER_OUTGOING_PACKETS :: 128
NET_SERVER_COMMAND_LOG_INTERVAL :: u32(60)
NET_SERVER_TICK_HZ :: 64
NET_SERVER_TICK_TIME :: f32(1.0 / f32(NET_SERVER_TICK_HZ))
NET_SNAPSHOT_HZ :: 32
NET_SNAPSHOT_INTERVAL_TICKS :: u32(NET_SERVER_TICK_HZ / NET_SNAPSHOT_HZ)

NetServerPeer :: distinct uintptr

NetServerSession :: struct {
	peer:     NetServerPeer,
	active:   bool,
	accepted: bool,
	player_id: u32,
	pending_cmds: [NET_SERVER_PENDING_USER_CMDS]protocol.User_Cmd,
	pending_cmd_count: int,
	last_input_cmd: protocol.User_Cmd,
	has_last_input_cmd: bool,
	last_processed_cmd_sequence: u32,
	last_logged_command: u32,
}

NetServerOutgoingPacket :: struct {
	peer:    NetServerPeer,
	channel: u8,
	mode:    transport.Packet_Mode,
	length:  int,
	data:    [protocol.MAX_PACKET_SIZE]byte,
}

NetServer :: struct {
	sessions: [NET_SERVER_MAX_CLIENTS]NetServerSession,
	scene:    ^Scene,
	map_name: string,
	content_id: u32,
	server_tick: u32,
	snapshot_sequence: u32,
	outgoing: [NET_SERVER_OUTGOING_PACKETS]NetServerOutgoingPacket,
	outgoing_get: u32,
	outgoing_send: u32,
}

net_server_create :: proc(scene: ^Scene, map_name: string, content_id: u32) -> NetServer {
	return {scene = scene, map_name = map_name, content_id = content_id}
}

net_server_connect :: proc(server: ^NetServer, peer: NetServerPeer) {
	log.infof("Client connected peer=%v", peer)
	net_server_add_session(server, peer)
}

net_server_disconnect :: proc(server: ^NetServer, peer: NetServerPeer) {
	log.infof("Client disconnected peer=%v", peer)
	net_server_remove_session(server, peer)
}

net_server_handle_packet :: proc(server: ^NetServer, peer: NetServerPeer, channel: u8, data: []byte, truncated := false) {
	if truncated {
		log.warnf("Dropped truncated packet peer=%v channel=%d bytes=%d", peer, channel, len(data))
		return
	}

	packet, err := protocol.parse_packet(data, channel)
	if err != .None {
		log.warnf("Rejected packet peer=%v channel=%d bytes=%d err=%v", peer, channel, len(data), err)
		return
	}

	#partial switch packet.header.kind {
	case .Client_Hello:
		hello := packet.hello
		hello_map := hello.map_name
		requested_map := protocol.map_name_string(&hello_map)
		log.infof("Client hello peer=%v protocol=%d requested_map=%s content_id=%08x", peer, packet.header.version, requested_map, hello.content_id)
		if reason := net_server_validate_client_hello(server, hello); reason != .None {
			net_server_log_reject_reason(peer, hello, server, reason)
			net_server_send_reject(server, peer, reason)
			return
		}
		player_id, accepted := net_server_accept_session(server, peer)
		if !accepted {
			log.warnf("Failed to accept peer=%v", peer)
			return
		}

		buffer: [protocol.MAX_PACKET_SIZE]byte
		response, ok := protocol.write_server_hello(buffer[:], server.map_name, server.content_id, player_id)
		if !ok || !net_server_queue_packet(server, peer, protocol.CHANNEL_CONTROL, response, .Reliable) {
			log.warnf("Failed to send server hello peer=%v", peer)
		}
	case .User_Cmd:
		index := net_server_find_session(server, peer)
		if index < 0 || !server.sessions[index].accepted {
			log.warnf("Dropped user cmd from unaccepted peer=%v", peer)
			return
		}
		net_server_enqueue_user_cmds(server, index, packet.user_cmds)
		if packet.user_cmds.count > 0 {
			net_server_log_user_cmd_if_needed(server, index, peer, packet.user_cmds.cmds[int(packet.user_cmds.count) - 1])
		}
	case:
		log.warnf("Unexpected packet kind peer=%v kind=%v", peer, packet.header.kind)
	}
}

net_server_tick :: proc(server: ^NetServer) {
	inputs: [NET_SERVER_MAX_CLIENTS]PlayerTickInput
	input_count := 0
	for &session in server.sessions {
		if !session.active || !session.accepted {
			continue
		}
		player := scene_player(server.scene, session.player_id)
		if player == nil {
			continue
		}
		input := PlayerTickInput{player_id = session.player_id, yaw = player.yaw, pitch = player.pitch}
		cmd, has_cmd := net_server_consume_user_cmds(&session)
		if has_cmd {
			input.move = player_move_input_from_user_cmd(cmd)
			input.yaw = cmd.yaw
			input.pitch = cmd.pitch
		} else if session.has_last_input_cmd {
			input.move = player_move_input_from_user_cmd(session.last_input_cmd)
			input.yaw = session.last_input_cmd.yaw
			input.pitch = session.last_input_cmd.pitch
		}
		inputs[input_count] = input
		input_count += 1
	}

	server.server_tick += 1
	scene_fixed_update_players(server.scene, inputs[:input_count], NET_SERVER_TICK_TIME)
	if server.server_tick % NET_SNAPSHOT_INTERVAL_TICKS == 0 {
		net_server_broadcast_snapshot(server)
	}
}

net_server_poll_outgoing :: proc(server: ^NetServer) -> (packet: NetServerOutgoingPacket, ok: bool) {
	if server.outgoing_send - server.outgoing_get > NET_SERVER_OUTGOING_PACKETS {
		server.outgoing_get = server.outgoing_send - NET_SERVER_OUTGOING_PACKETS
	}
	if server.outgoing_get >= server.outgoing_send {
		return {}, false
	}
	index := server.outgoing_get % NET_SERVER_OUTGOING_PACKETS
	server.outgoing_get += 1
	return server.outgoing[index], true
}

net_server_queue_packet :: proc(server: ^NetServer, peer: NetServerPeer, channel: u8, data: []byte, mode: transport.Packet_Mode) -> bool {
	if len(data) > protocol.MAX_PACKET_SIZE {
		log.warnf("Dropped oversized server packet peer=%v channel=%d bytes=%d", peer, channel, len(data))
		return false
	}
	if server.outgoing_send - server.outgoing_get >= NET_SERVER_OUTGOING_PACKETS {
		log.warn("Dropping oldest server packet: outgoing queue is full")
		server.outgoing_get += 1
	}
	index := server.outgoing_send % NET_SERVER_OUTGOING_PACKETS
	packet := &server.outgoing[index]
	packet.peer = peer
	packet.channel = channel
	packet.mode = mode
	packet.length = len(data)
	copy(packet.data[:], data[:packet.length])
	server.outgoing_send += 1
	return true
}

net_server_add_session :: proc(server: ^NetServer, peer: NetServerPeer) {
	if index := net_server_find_session(server, peer); index >= 0 {
		server.sessions[index].accepted = false
		server.sessions[index].active = true
		server.sessions[index].peer = peer
		server.sessions[index].player_id = u32(index + 1)
		return
	}
	for i in 0..<len(server.sessions) {
		if !server.sessions[i].active {
			server.sessions[i] = NetServerSession{
				peer = peer,
				active = true,
				player_id = u32(i + 1),
			}
			return
		}
	}
	log.warnf("No free server session slot for peer=%v", peer)
}

net_server_remove_session :: proc(server: ^NetServer, peer: NetServerPeer) {
	if index := net_server_find_session(server, peer); index >= 0 {
		server.sessions[index] = {}
	}
}

net_server_accept_session :: proc(server: ^NetServer, peer: NetServerPeer) -> (player_id: u32, ok: bool) {
	if index := net_server_find_session(server, peer); index >= 0 {
		session := &server.sessions[index]
		scene_reset_player_to_spawn(server.scene, session.player_id)
		session.accepted = true
		return session.player_id, true
	}
	return 0, false
}

net_server_find_session :: proc(server: ^NetServer, peer: NetServerPeer) -> int {
	for i in 0..<len(server.sessions) {
		if server.sessions[i].active && server.sessions[i].peer == peer {
			return i
		}
	}
	return -1
}

net_server_log_user_cmd_if_needed :: proc(server: ^NetServer, session_index: int, peer: NetServerPeer, cmd: protocol.User_Cmd) {
	session := &server.sessions[session_index]
	if cmd.sequence - session.last_logged_command < NET_SERVER_COMMAND_LOG_INTERVAL {
		return
	}
	session.last_logged_command = cmd.sequence
	player := scene_player(server.scene, session.player_id)
	if player == nil {
		return
	}
	position := player.position
	log.debugf("User cmd peer=%v player=%d seq=%d tick=%d move=(%.2f, %.2f) pos=(%.2f, %.2f, %.2f) yaw=%.3f pitch=%.3f buttons=%04x", peer, session.player_id, cmd.sequence, cmd.client_tick, cmd.move_forward, cmd.move_right, position.x, position.y, position.z, player.yaw, player.pitch, cmd.buttons)
}

net_server_broadcast_snapshot :: proc(server: ^NetServer) {
	buffer: [protocol.MAX_PACKET_SIZE]byte
	base_snapshot := protocol.Server_Snapshot{server_tick = server.server_tick, sequence = server.snapshot_sequence}
	for session in server.sessions {
		if !session.active || !session.accepted {
			continue
		}
		if base_snapshot.player_count >= protocol.MAX_SNAPSHOT_PLAYERS {
			break
		}
		player := scene_player(server.scene, session.player_id)
		if player == nil {
			continue
		}
		position := player.position
		base_snapshot.players[base_snapshot.player_count] = protocol.Server_Player_State{
			player_id = session.player_id,
			position = position,
			velocity = player.velocity,
			yaw = player.yaw,
			pitch = player.pitch,
			grounded = player.grounded,
			ground_normal = player.ground_normal,
		}
		base_snapshot.player_count += 1
	}

	for session in server.sessions {
		if !session.active || !session.accepted {
			continue
		}
		snapshot := base_snapshot
		snapshot.last_processed_user_cmd = session.last_processed_cmd_sequence
		packet, ok := protocol.write_server_snapshot(buffer[:], snapshot)
		if !ok {
			log.warnf("Failed to write server snapshot tick=%d players=%d", snapshot.server_tick, snapshot.player_count)
			return
		}
		if !net_server_queue_packet(server, session.peer, protocol.CHANNEL_SNAPSHOTS, packet, .Unreliable) {
			log.warnf("Failed to queue server snapshot tick=%d peer=%v", snapshot.server_tick, session.peer)
		}
	}
	server.snapshot_sequence += 1
}

net_server_enqueue_user_cmds :: proc(server: ^NetServer, session_index: int, cmds: protocol.User_Cmds) {
	session := &server.sessions[session_index]
	for i in 0..<int(cmds.count) {
		net_server_enqueue_user_cmd(session, cmds.cmds[i])
	}
}

net_server_enqueue_user_cmd :: proc(session: ^NetServerSession, cmd: protocol.User_Cmd) {
	if cmd.sequence <= session.last_processed_cmd_sequence {
		return
	}
	for i in 0..<session.pending_cmd_count {
		if session.pending_cmds[i].sequence == cmd.sequence {
			return
		}
	}

	if session.pending_cmd_count >= NET_SERVER_PENDING_USER_CMDS {
		copy(session.pending_cmds[:], session.pending_cmds[1:])
		session.pending_cmd_count -= 1
	}

	insert_at := session.pending_cmd_count
	for insert_at > 0 && session.pending_cmds[insert_at - 1].sequence > cmd.sequence {
		session.pending_cmds[insert_at] = session.pending_cmds[insert_at - 1]
		insert_at -= 1
	}
	session.pending_cmds[insert_at] = cmd
	session.pending_cmd_count += 1
}

net_server_consume_user_cmds :: proc(session: ^NetServerSession) -> (cmd: protocol.User_Cmd, ok: bool) {
	if session.pending_cmd_count == 0 {
		return {}, false
	}

	cmd = session.pending_cmds[0]
	if session.pending_cmd_count > 1 {
		copy(session.pending_cmds[:], session.pending_cmds[1:session.pending_cmd_count])
	}
	session.pending_cmd_count -= 1
	session.last_processed_cmd_sequence = cmd.sequence
	session.last_input_cmd = cmd
	session.has_last_input_cmd = true
	return cmd, true
}

net_server_validate_client_hello :: proc(server: ^NetServer, hello: protocol.Hello_Payload) -> protocol.Reject_Reason {
	hello_map := hello.map_name
	if !protocol.map_name_equals_string(&hello_map, server.map_name) {
		return .Map_Mismatch
	}
	if hello.content_id != server.content_id {
		return .Content_Mismatch
	}
	return .None
}

net_server_send_reject :: proc(server: ^NetServer, peer: NetServerPeer, reason: protocol.Reject_Reason) {
	buffer: [protocol.MAX_PACKET_SIZE]byte
	packet, ok := protocol.write_server_reject(buffer[:], reason)
	if !ok || !net_server_queue_packet(server, peer, protocol.CHANNEL_CONTROL, packet, .Reliable) {
		log.warnf("Failed to send server reject peer=%v reason=%v", peer, reason)
	}
}

net_server_log_reject_reason :: proc(peer: NetServerPeer, hello: protocol.Hello_Payload, server: ^NetServer, reason: protocol.Reject_Reason) {
	#partial switch reason {
	case .Map_Mismatch:
		hello_map := hello.map_name
		log.warnf("Rejecting peer=%v for map mismatch requested=%s server=%s", peer, protocol.map_name_string(&hello_map), server.map_name)
	case .Content_Mismatch:
		log.warnf("Rejecting peer=%v for content mismatch requested=%08x server=%08x", peer, hello.content_id, server.content_id)
	case:
		log.warnf("Rejecting peer=%v reason=%v", peer, reason)
	}
}

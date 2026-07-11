package main

import "core:fmt"
import "core:log"
import "core:os"
import "core:strings"
import flags "core:flags"
import "core:time"
import engine "../engine"
import game "../game"
import protocol "../protocol"
import transport "../net"

SERVER_DEFAULT_PORT :: u16(29001)
SERVER_MAX_CLIENTS  :: 32
SERVER_DEFAULT_MAP   :: "test"
SERVER_DEFAULT_CONTENT_ID :: u32(0)
SERVER_PENDING_USER_CMDS :: 64

ClientSession :: struct {
	peer:     transport.Peer,
	active:   bool,
	accepted: bool,
	player_id: u32,
	pending_cmds: [SERVER_PENDING_USER_CMDS]protocol.User_Cmd,
	pending_cmd_count: int,
	last_input_cmd: protocol.User_Cmd,
	has_last_input_cmd: bool,
	last_processed_cmd_sequence: u32,
	last_logged_command: u32,
}

ServerState :: struct {
	sessions: [SERVER_MAX_CLIENTS]ClientSession,
	scene:    ^game.Scene,
	server_tick: u32,
	snapshot_sequence: u32,
}

SERVER_COMMAND_LOG_INTERVAL :: u32(60)
SERVER_TICK_TIME :: f32(1.0 / 40.0)

Options :: struct {
	base_dir:   string `args:"name=basedir" usage:"Base directory containing game content."`,
	game_dir:   string `args:"name=game" usage:"Game directory to search before base."`,
	port:       u16    `usage:"UDP port to listen on."`,
	map_name:   string `args:"name=map" usage:"Authoritative map name for handshake validation."`,
	content_id: u32    `usage:"Authoritative map/content identifier for handshake validation."`,
	seconds:    f32    `usage:"Optional run duration for smoke tests. Zero runs until interrupted."`,
}

main :: proc() {
	context.logger = log.create_console_logger()
	options := parse_options(os.args[1:])
	if options.port == 0 {
		options.port = SERVER_DEFAULT_PORT
	}

	host, ok := transport.server_create(options.port, uint(SERVER_MAX_CLIENTS))
	if !ok {
		log.fatalf("Failed to start ENet server on port %d", options.port)
	}
	defer transport.destroy(&host)
	fs := engine.game_fs_create(options.base_dir, options.game_dir)
	defer engine.game_fs_destroy(&fs)
	map_qpath := server_map_qpath(options.map_name)
	defer delete(map_qpath)
	assets := game.scene_assets_load(&fs, map_qpath)
	defer game.scene_assets_destroy(&assets)

	log.infof("Dedicated server listening on UDP port %d map=%s content_id=%08x", options.port, options.map_name, options.content_id)
	scene := game.scene_create(&assets, {})
	defer game.scene_destroy(&scene)

	run_server(&host, options, &scene)
}

parse_options :: proc(args: []string) -> Options {
	runtime_args := make([dynamic]string, 0, len(args) + 1, context.temp_allocator)
	append(&runtime_args, "arctic-char-server")
	for arg in args {
		append(&runtime_args, arg)
	}

	options := Options{base_dir = ".", port = SERVER_DEFAULT_PORT, map_name = SERVER_DEFAULT_MAP, content_id = SERVER_DEFAULT_CONTENT_ID}
	flags.parse_or_exit(&options, runtime_args[:], .Unix)
	return options
}

server_map_qpath :: proc(map_name: string, allocator := context.allocator) -> string {
	qpath, err := strings.concatenate({"maps/", map_name, ".map"}, allocator)
	assert(err == nil)
	return qpath
}

run_server :: proc(host: ^transport.Host, options: Options, scene: ^game.Scene) {
	receive_buffer: [protocol.MAX_PACKET_SIZE]byte
	send_buffer: [protocol.MAX_PACKET_SIZE]byte
	state := ServerState{scene = scene}
	started := time.now()
	last_tick_time := started
	accumulator := f32(0)
	for {
		if options.seconds > 0 {
			elapsed := time.duration_seconds(time.since(started))
			if elapsed >= f64(options.seconds) {
				log.info("Dedicated server smoke duration elapsed")
				return
			}
		}

		for _ in 0..<32 {
			event := transport.poll(host, receive_buffer[:], 1)
			if event.kind == .None {
				break
			}
			server_handle_event(host, &state, options, event, send_buffer[:])
		}

		frame_time := f32(time.duration_seconds(time.since(last_tick_time)))
		last_tick_time = time.now()
		accumulator += min(frame_time, 0.25)
		for accumulator >= SERVER_TICK_TIME {
			server_tick(host, &state, send_buffer[:])
			accumulator -= SERVER_TICK_TIME
		}
	}
}

server_handle_event :: proc(host: ^transport.Host, state: ^ServerState, options: Options, event: transport.Event, send_buffer: []byte) {
	#partial switch event.kind {
	case .Connect:
		log.infof("Client connected peer=%v rtt=%dms", event.peer, transport.peer_round_trip_ms(event.peer))
		server_add_session(state, event.peer)
	case .Disconnect:
		log.infof("Client disconnected peer=%v", event.peer)
		server_remove_session(state, event.peer)
	case .Receive:
		if event.truncated {
			log.warnf("Dropped truncated packet peer=%v channel=%d bytes=%d", event.peer, event.channel, len(event.data))
			return
		}

		packet, err := protocol.parse_packet(event.data, event.channel)
		if err != .None {
			log.warnf("Rejected packet peer=%v channel=%d bytes=%d err=%v", event.peer, event.channel, len(event.data), err)
			return
		}

		#partial switch packet.header.kind {
		case .Client_Hello:
			hello := packet.hello
			hello_map := hello.map_name
			requested_map := protocol.map_name_string(&hello_map)
			log.infof("Client hello peer=%v protocol=%d requested_map=%s content_id=%08x", event.peer, packet.header.version, requested_map, hello.content_id)
			if reason := validate_client_hello(hello, options); reason != .None {
				log_reject_reason(event.peer, hello, options, reason)
				send_reject(host, event.peer, send_buffer, reason)
				return
			}
			player_id, accepted := server_accept_session(state, event.peer)
			if !accepted {
				log.warnf("Failed to accept peer=%v", event.peer)
				return
			}

			response, ok := protocol.write_server_hello(send_buffer, options.map_name, options.content_id, player_id)
			if !ok || !transport.send(event.peer, protocol.CHANNEL_CONTROL, response, .Reliable) {
				log.warnf("Failed to send server hello peer=%v", event.peer)
			}
			transport.flush(host)
		case .User_Cmd:
			index := server_find_session(state, event.peer)
			if index < 0 || !state.sessions[index].accepted {
				log.warnf("Dropped user cmd from unaccepted peer=%v", event.peer)
				return
			}
			server_enqueue_user_cmds(state, index, packet.user_cmds)
			if packet.user_cmds.count > 0 {
				server_log_user_cmd_if_needed(state, index, event.peer, packet.user_cmds.cmds[int(packet.user_cmds.count) - 1])
			}
		case:
			log.warnf("Unexpected packet kind peer=%v kind=%v", event.peer, packet.header.kind)
		}
	case:
	}
}

server_add_session :: proc(state: ^ServerState, peer: transport.Peer) {
	if index := server_find_session(state, peer); index >= 0 {
		state.sessions[index].accepted = false
		state.sessions[index].active = true
		state.sessions[index].peer = peer
		state.sessions[index].player_id = u32(index + 1)
		return
	}
	for i in 0..<len(state.sessions) {
		if !state.sessions[i].active {
			state.sessions[i] = ClientSession{
				peer = peer,
				active = true,
				player_id = u32(i + 1),
			}
			return
		}
	}
	log.warnf("No free server session slot for peer=%v", peer)
}

server_remove_session :: proc(state: ^ServerState, peer: transport.Peer) {
	if index := server_find_session(state, peer); index >= 0 {
		state.sessions[index] = {}
	}
}

server_accept_session :: proc(state: ^ServerState, peer: transport.Peer) -> (player_id: u32, ok: bool) {
	if index := server_find_session(state, peer); index >= 0 {
		session := &state.sessions[index]
		game.scene_reset_player_to_spawn(state.scene, session.player_id)
		session.accepted = true
		return session.player_id, true
	}
	return 0, false
}

server_find_session :: proc(state: ^ServerState, peer: transport.Peer) -> int {
	for i in 0..<len(state.sessions) {
		if state.sessions[i].active && state.sessions[i].peer == peer {
			return i
		}
	}
	return -1
}

server_log_user_cmd_if_needed :: proc(state: ^ServerState, session_index: int, peer: transport.Peer, cmd: protocol.User_Cmd) {
	session := &state.sessions[session_index]
	if cmd.sequence - session.last_logged_command < SERVER_COMMAND_LOG_INTERVAL {
		return
	}
	session.last_logged_command = cmd.sequence
	player := game.scene_player(state.scene, session.player_id)
	if player == nil {
		return
	}
	position := player.position
	log.debugf("User cmd peer=%v player=%d seq=%d tick=%d move=(%.2f, %.2f) pos=(%.2f, %.2f, %.2f) yaw=%.3f pitch=%.3f buttons=%04x", peer, session.player_id, cmd.sequence, cmd.client_tick, cmd.move_forward, cmd.move_right, position.x, position.y, position.z, player.yaw, player.pitch, cmd.buttons)
}

server_tick :: proc(host: ^transport.Host, state: ^ServerState, buffer: []byte) {
	inputs: [SERVER_MAX_CLIENTS]game.PlayerTickInput
	input_count := 0
	for &session in state.sessions {
		if !session.active || !session.accepted {
			continue
		}
		player := game.scene_player(state.scene, session.player_id)
		if player == nil {
			continue
		}
		input := game.PlayerTickInput{player_id = session.player_id, yaw = player.yaw, pitch = player.pitch}
		cmd, has_cmd := server_consume_user_cmds(&session)
		if has_cmd {
			input.move = game.player_move_input_from_user_cmd(cmd)
			input.yaw = cmd.yaw
			input.pitch = cmd.pitch
		} else if session.has_last_input_cmd {
			input.move = game.player_move_input_from_user_cmd(session.last_input_cmd)
			input.yaw = session.last_input_cmd.yaw
			input.pitch = session.last_input_cmd.pitch
		}
		inputs[input_count] = input
		input_count += 1
	}

	if input_count == 0 {
		return
	}

	state.server_tick += 1
	game.scene_fixed_update_players(state.scene, inputs[:input_count], SERVER_TICK_TIME)
	server_broadcast_snapshot(host, state, buffer)
}

server_broadcast_snapshot :: proc(host: ^transport.Host, state: ^ServerState, buffer: []byte) {
	base_snapshot := protocol.Server_Snapshot{server_tick = state.server_tick, sequence = state.snapshot_sequence}
	for session in state.sessions {
		if !session.active || !session.accepted {
			continue
		}
		if base_snapshot.player_count >= protocol.MAX_SNAPSHOT_PLAYERS {
			break
		}
		player := game.scene_player(state.scene, session.player_id)
		if player == nil {
			continue
		}
		position := player.position
		base_snapshot.players[base_snapshot.player_count] = protocol.Server_Player_State{
			player_id = session.player_id,
			position = position,
			yaw = player.yaw,
		}
		base_snapshot.player_count += 1
	}

	for session in state.sessions {
		if !session.active || !session.accepted {
			continue
		}
		snapshot := base_snapshot
		snapshot.last_processed_user_cmd = session.last_processed_cmd_sequence
		packet, ok := protocol.write_server_snapshot(buffer, snapshot)
		if !ok {
			log.warnf("Failed to write server snapshot tick=%d players=%d", snapshot.server_tick, snapshot.player_count)
			return
		}
		if !transport.send(session.peer, protocol.CHANNEL_SNAPSHOTS, packet, .Unreliable) {
			log.warnf("Failed to send server snapshot tick=%d to peer=%v", snapshot.server_tick, session.peer)
		}
	}
	state.snapshot_sequence += 1
	transport.flush(host)
}

server_enqueue_user_cmds :: proc(state: ^ServerState, session_index: int, cmds: protocol.User_Cmds) {
	session := &state.sessions[session_index]
	for i in 0..<int(cmds.count) {
		server_enqueue_user_cmd(session, cmds.cmds[i])
	}
}

server_enqueue_user_cmd :: proc(session: ^ClientSession, cmd: protocol.User_Cmd) {
	if cmd.sequence <= session.last_processed_cmd_sequence {
		return
	}
	for i in 0..<session.pending_cmd_count {
		if session.pending_cmds[i].sequence == cmd.sequence {
			return
		}
	}

	if session.pending_cmd_count >= SERVER_PENDING_USER_CMDS {
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

server_consume_user_cmds :: proc(session: ^ClientSession) -> (cmd: protocol.User_Cmd, ok: bool) {
	if session.pending_cmd_count == 0 {
		return {}, false
	}

	for i in 0..<session.pending_cmd_count {
		cmd = session.pending_cmds[i]
		session.last_processed_cmd_sequence = cmd.sequence
	}
	session.pending_cmd_count = 0
	session.last_input_cmd = cmd
	session.has_last_input_cmd = true
	return cmd, true
}

validate_client_hello :: proc(hello: protocol.Hello_Payload, options: Options) -> protocol.Reject_Reason {
	hello_map := hello.map_name
	if !protocol.map_name_equals_string(&hello_map, options.map_name) {
		return .Map_Mismatch
	}
	if hello.content_id != options.content_id {
		return .Content_Mismatch
	}
	return .None
}

send_reject :: proc(host: ^transport.Host, peer: transport.Peer, buffer: []byte, reason: protocol.Reject_Reason) {
	packet, ok := protocol.write_server_reject(buffer, reason)
	if !ok || !transport.send(peer, protocol.CHANNEL_CONTROL, packet, .Reliable) {
		log.warnf("Failed to send server reject peer=%v reason=%v", peer, reason)
		return
	}
	transport.flush(host)
}

log_reject_reason :: proc(peer: transport.Peer, hello: protocol.Hello_Payload, options: Options, reason: protocol.Reject_Reason) {
	#partial switch reason {
	case .Map_Mismatch:
		hello_map := hello.map_name
		log.warnf("Rejecting peer=%v for map mismatch requested=%s server=%s", peer, protocol.map_name_string(&hello_map), options.map_name)
	case .Content_Mismatch:
		log.warnf("Rejecting peer=%v for content mismatch requested=%08x server=%08x", peer, hello.content_id, options.content_id)
	case:
		log.warnf("Rejecting peer=%v reason=%v", peer, reason)
	}
}

when ODIN_OS == .Windows {
	@(export)
	NvOptimusEnablement: u32 = 1

	@(export)
	AmdPowerXpressRequestHighPerformance: u32 = 1
}

package main

import "core:fmt"
import "core:log"
import "core:math"
import "core:os"
import flags "core:flags"
import "core:time"
import protocol "../protocol"
import transport "../net"

SERVER_DEFAULT_PORT :: u16(29001)
SERVER_MAX_CLIENTS  :: 32
SERVER_DEFAULT_MAP   :: "test"
SERVER_DEFAULT_CONTENT_ID :: u32(0)

ClientSession :: struct {
	peer:     transport.Peer,
	active:   bool,
	accepted: bool,
	player_id: u32,
	position: [3]f32,
	yaw:      f32,
	last_logged_command: u32,
}

ServerState :: struct {
	sessions: [SERVER_MAX_CLIENTS]ClientSession,
}

SERVER_PLAYER_START :: [3]f32{0, 0.9, 8.0}
SERVER_PLAYER_SPEED :: f32(5.0)
SERVER_COMMAND_STEP :: f32(1.0 / 60.0)
SERVER_COMMAND_LOG_INTERVAL :: u32(60)

Options :: struct {
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

	log.infof("Dedicated server listening on UDP port %d map=%s content_id=%08x", options.port, options.map_name, options.content_id)
	run_server(&host, options)
}

parse_options :: proc(args: []string) -> Options {
	runtime_args := make([dynamic]string, 0, len(args) + 1, context.temp_allocator)
	append(&runtime_args, "arctic-char-server")
	for arg in args {
		append(&runtime_args, arg)
	}

	options := Options{port = SERVER_DEFAULT_PORT, map_name = SERVER_DEFAULT_MAP, content_id = SERVER_DEFAULT_CONTENT_ID}
	flags.parse_or_exit(&options, runtime_args[:], .Unix)
	return options
}

run_server :: proc(host: ^transport.Host, options: Options) {
	receive_buffer: [protocol.MAX_PACKET_SIZE]byte
	send_buffer: [protocol.MAX_PACKET_SIZE]byte
	state: ServerState
	started := time.now()
	for {
		if options.seconds > 0 {
			elapsed := time.duration_seconds(time.since(started))
			if elapsed >= f64(options.seconds) {
				log.info("Dedicated server smoke duration elapsed")
				return
			}
		}

		event := transport.poll(host, receive_buffer[:], 16)
		switch event.kind {
		case .None:
			// No event this tick.
		case .Connect:
			log.infof("Client connected peer=%v rtt=%dms", event.peer, transport.peer_round_trip_ms(event.peer))
			server_add_session(&state, event.peer)
		case .Disconnect:
			log.infof("Client disconnected peer=%v", event.peer)
			server_remove_session(&state, event.peer)
		case .Receive:
			if event.truncated {
				log.warnf("Dropped truncated packet peer=%v channel=%d bytes=%d", event.peer, event.channel, len(event.data))
				continue
			}

			packet, err := protocol.parse_packet(event.data, event.channel)
			if err != .None {
				log.warnf("Rejected packet peer=%v channel=%d bytes=%d err=%v", event.peer, event.channel, len(event.data), err)
				continue
			}

			#partial switch packet.header.kind {
			case .Client_Hello:
				hello := packet.hello
				hello_map := hello.map_name
				requested_map := protocol.map_name_string(&hello_map)
				log.infof("Client hello peer=%v protocol=%d requested_map=%s content_id=%08x", event.peer, packet.header.version, requested_map, hello.content_id)
				if reason := validate_client_hello(hello, options); reason != .None {
					log_reject_reason(event.peer, hello, options, reason)
					send_reject(host, event.peer, send_buffer[:], reason)
					continue
				}
				server_accept_session(&state, event.peer)

				response, ok := protocol.write_server_hello(send_buffer[:], options.map_name, options.content_id)
				if !ok || !transport.send(event.peer, protocol.CHANNEL_CONTROL, response, .Reliable) {
					log.warnf("Failed to send server hello peer=%v", event.peer)
				}
				transport.flush(host)
			case .User_Cmd:
				index := server_find_session(&state, event.peer)
				if index < 0 || !state.sessions[index].accepted {
					log.warnf("Dropped user cmd from unaccepted peer=%v", event.peer)
					continue
				}
				server_apply_user_cmd(&state.sessions[index], packet.user_cmd)
				server_log_user_cmd_if_needed(&state.sessions[index], event.peer, packet.user_cmd)
				server_broadcast_player_state(host, &state, index, send_buffer[:])
			case:
				log.warnf("Unexpected packet kind peer=%v kind=%v", event.peer, packet.header.kind)
			}
		}
	}
}

server_add_session :: proc(state: ^ServerState, peer: transport.Peer) {
	if index := server_find_session(state, peer); index >= 0 {
		state.sessions[index].accepted = false
		return
	}
	for i in 0..<len(state.sessions) {
		if !state.sessions[i].active {
			state.sessions[i] = ClientSession{
				peer = peer,
				active = true,
				player_id = u32(i + 1),
				position = SERVER_PLAYER_START,
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

server_accept_session :: proc(state: ^ServerState, peer: transport.Peer) {
	if index := server_find_session(state, peer); index >= 0 {
		state.sessions[index].accepted = true
	}
}

server_session_accepted :: proc(state: ^ServerState, peer: transport.Peer) -> bool {
	if index := server_find_session(state, peer); index >= 0 {
		return state.sessions[index].accepted
	}
	return false
}

server_find_session :: proc(state: ^ServerState, peer: transport.Peer) -> int {
	for i in 0..<len(state.sessions) {
		if state.sessions[i].active && state.sessions[i].peer == peer {
			return i
		}
	}
	return -1
}

server_apply_user_cmd :: proc(session: ^ClientSession, cmd: protocol.User_Cmd) {
	session.yaw = cmd.yaw
	forward := [3]f32{math.sin(cmd.yaw), 0, -math.cos(cmd.yaw)}
	right := [3]f32{math.cos(cmd.yaw), 0, math.sin(cmd.yaw)}
	dx := (forward.x * cmd.move_forward + right.x * cmd.move_right) * SERVER_PLAYER_SPEED * SERVER_COMMAND_STEP
	dz := (forward.z * cmd.move_forward + right.z * cmd.move_right) * SERVER_PLAYER_SPEED * SERVER_COMMAND_STEP
	session.position.x += dx
	session.position.z += dz
}

server_log_user_cmd_if_needed :: proc(session: ^ClientSession, peer: transport.Peer, cmd: protocol.User_Cmd) {
	if cmd.sequence - session.last_logged_command < SERVER_COMMAND_LOG_INTERVAL {
		return
	}
	session.last_logged_command = cmd.sequence
	log.debugf("User cmd peer=%v player=%d seq=%d tick=%d move=(%.2f, %.2f) pos=(%.2f, %.2f, %.2f) yaw=%.3f pitch=%.3f buttons=%04x", peer, session.player_id, cmd.sequence, cmd.client_tick, cmd.move_forward, cmd.move_right, session.position.x, session.position.y, session.position.z, cmd.yaw, cmd.pitch, cmd.buttons)
}

server_broadcast_player_state :: proc(host: ^transport.Host, state: ^ServerState, source_index: int, buffer: []byte) {
	source := state.sessions[source_index]
	player_state := protocol.Server_Player_State{
		player_id = source.player_id,
		position = source.position,
		yaw = source.yaw,
	}
	packet, ok := protocol.write_server_player_state(buffer, player_state)
	if !ok {
		log.warnf("Failed to write player state player=%d", source.player_id)
		return
	}

	for i in 0..<len(state.sessions) {
		if i == source_index || !state.sessions[i].active || !state.sessions[i].accepted {
			continue
		}
		if !transport.send(state.sessions[i].peer, protocol.CHANNEL_SNAPSHOTS, packet, .Unreliable) {
			log.warnf("Failed to send player state player=%d to peer=%v", source.player_id, state.sessions[i].peer)
		}
	}
	transport.flush(host)
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

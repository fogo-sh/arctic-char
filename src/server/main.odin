package main

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
	server := new(game.NetServer)
	defer free(server)
	server^ = game.net_server_create(scene, options.map_name, options.content_id)
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
			server_handle_event(server, event)
		}

		frame_time := f32(time.duration_seconds(time.since(last_tick_time)))
		last_tick_time = time.now()
		accumulator += min(frame_time, 0.25)
		for accumulator >= game.NET_SERVER_TICK_TIME {
			game.net_server_tick(server)
			server_flush_outgoing(host, server)
			accumulator -= game.NET_SERVER_TICK_TIME
		}
	}
}

server_handle_event :: proc(server: ^game.NetServer, event: transport.Event) {
	#partial switch event.kind {
	case .Connect:
		log.infof("ENet peer connected peer=%v rtt=%dms", event.peer, transport.peer_round_trip_ms(event.peer))
		game.net_server_connect(server, server_peer_from_transport(event.peer))
	case .Disconnect:
		game.net_server_disconnect(server, server_peer_from_transport(event.peer))
	case .Receive:
		game.net_server_handle_packet(server, server_peer_from_transport(event.peer), event.channel, event.data, event.truncated)
	case:
	}
}

server_flush_outgoing :: proc(host: ^transport.Host, server: ^game.NetServer) {
	flushed := false
	for {
		packet, ok := game.net_server_poll_outgoing(server)
		if !ok {
			break
		}
		peer := transport_peer_from_server(packet.peer)
		if !transport.send(peer, packet.channel, packet.data[:packet.length], packet.mode) {
			log.warnf("Failed to send server packet channel=%d to peer=%v", packet.channel, peer)
			continue
		}
		flushed = true
	}
	if flushed {
		transport.flush(host)
	}
}

server_peer_from_transport :: proc(peer: transport.Peer) -> game.NetServerPeer {
	return game.NetServerPeer(uintptr(peer))
}

transport_peer_from_server :: proc(peer: game.NetServerPeer) -> transport.Peer {
	return transport.Peer(uintptr(peer))
}

when ODIN_OS == .Windows {
	@(export)
	NvOptimusEnablement: u32 = 1

	@(export)
	AmdPowerXpressRequestHighPerformance: u32 = 1
}

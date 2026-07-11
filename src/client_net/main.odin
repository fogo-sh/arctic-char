package main

import "core:log"
import "core:os"
import flags "core:flags"
import "core:time"
import protocol "../protocol"
import transport "../net"

CLIENT_DEFAULT_ADDRESS :: "127.0.0.1"
CLIENT_DEFAULT_PORT    :: u16(29001)
CLIENT_DEFAULT_MAP     :: "test"
CLIENT_DEFAULT_CONTENT_ID :: u32(0)

Options :: struct {
	address:    string `usage:"Server address to connect to."`,
	port:       u16    `usage:"Server UDP port."`,
	map_name:   string `args:"name=map" usage:"Requested map name."`,
	content_id: u32    `usage:"Requested map/content identifier."`,
	seconds:    f32    `usage:"Connection timeout/smoke duration."`,
}

main :: proc() {
	context.logger = log.create_console_logger()
	options := parse_options(os.args[1:])

	host, ok := transport.client_create()
	if !ok {
		log.fatal("Failed to create ENet client")
	}
	defer transport.destroy(&host)

	peer: transport.Peer
	peer, ok = transport.connect(&host, options.address, options.port)
	if !ok {
		log.fatalf("Failed to start connection to %s:%d", options.address, options.port)
	}

	log.infof("Connecting to %s:%d", options.address, options.port)
	if !run_client(&host, peer, options) {
		log.fatal("Client network smoke failed")
	}
}

parse_options :: proc(args: []string) -> Options {
	runtime_args := make([dynamic]string, 0, len(args) + 1, context.temp_allocator)
	append(&runtime_args, "arctic-char-client-net")
	for arg in args {
		append(&runtime_args, arg)
	}

	options := Options{address = CLIENT_DEFAULT_ADDRESS, port = CLIENT_DEFAULT_PORT, map_name = CLIENT_DEFAULT_MAP, content_id = CLIENT_DEFAULT_CONTENT_ID, seconds = 5}
	flags.parse_or_exit(&options, runtime_args[:], .Unix)
	return options
}

run_client :: proc(host: ^transport.Host, server_peer: transport.Peer, options: Options) -> bool {
	receive_buffer: [protocol.MAX_PACKET_SIZE]byte
	send_buffer: [protocol.MAX_PACKET_SIZE]byte
	started := time.now()
	hello_sent := false

	for {
		elapsed := time.duration_seconds(time.since(started))
		if elapsed >= f64(options.seconds) {
			log.warn("Timed out waiting for server hello")
			return false
		}

		event := transport.poll(host, receive_buffer[:], 16)
		switch event.kind {
		case .None:
			// No event this tick.
		case .Connect:
			log.infof("Connected peer=%v rtt=%dms", event.peer, transport.peer_round_trip_ms(event.peer))
			packet, ok := protocol.write_client_hello(send_buffer[:], options.map_name, options.content_id)
			if !ok || !transport.send(server_peer, protocol.CHANNEL_CONTROL, packet, .Reliable) {
				log.warn("Failed to send client hello")
				return false
			}
			transport.flush(host)
			hello_sent = true
		case .Disconnect:
			log.warnf("Disconnected peer=%v", event.peer)
			return false
		case .Receive:
			if event.truncated {
				log.warnf("Dropped truncated packet channel=%d bytes=%d", event.channel, len(event.data))
				continue
			}

			packet, err := protocol.parse_packet(event.data, event.channel)
			if err != .None {
				log.warnf("Rejected packet channel=%d bytes=%d err=%v", event.channel, len(event.data), err)
				continue
			}

			if packet.header.kind == .Server_Hello && hello_sent {
				hello := packet.hello
				hello_map := hello.map_name
				if !protocol.map_name_equals_string(&hello_map, options.map_name) || hello.content_id != options.content_id {
					log.warnf("Server hello mismatch map=%s content_id=%08x", protocol.map_name_string(&hello_map), hello.content_id)
					return false
				}
				log.infof("Server hello protocol=%d map=%s content_id=%08x", packet.header.version, protocol.map_name_string(&hello_map), hello.content_id)
				transport.disconnect(server_peer)
				transport.flush(host)
				return true
			}

			if packet.header.kind == .Server_Reject {
				log.warnf("Server rejected connection reason=%v", packet.reject_reason)
				return false
			}

			log.warnf("Unexpected packet kind=%v", packet.header.kind)
		}
	}
}

when ODIN_OS == .Windows {
	@(export)
	NvOptimusEnablement: u32 = 1

	@(export)
	AmdPowerXpressRequestHighPerformance: u32 = 1
}

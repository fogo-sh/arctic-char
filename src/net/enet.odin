package transport

import "core:log"
import "core:mem"
import "core:strings"
import enet "vendor:ENet"

DEFAULT_CHANNEL_COUNT :: uint(4)
DEFAULT_PORT          :: u16(29001)

Peer :: distinct uintptr

Host_Mode :: enum {
	None,
	Server,
	Client,
}

Packet_Mode :: enum {
	Unreliable,
	Reliable,
}

Event_Kind :: enum {
	None,
	Connect,
	Disconnect,
	Receive,
}

Host :: struct {
	handle: ^enet.Host,
	mode:   Host_Mode,
}

Event :: struct {
	kind:      Event_Kind,
	peer:      Peer,
	channel:   u8,
	data:      []byte,
	truncated: bool,
	user_data: u32,
}

initialized_count: int

initialize :: proc() -> bool {
	if initialized_count == 0 {
		if enet.initialize() != 0 {
			return false
		}
		version := enet.linked_version()
		log.debugf(
			"ENet initialized: linked=%d.%d.%d binding=%d.%d.%d",
			enet.VERSION_GET_MAJOR(version),
			enet.VERSION_GET_MINOR(version),
			enet.VERSION_GET_PATCH(version),
			enet.VERSION_MAJOR,
			enet.VERSION_MINOR,
			enet.VERSION_PATCH,
		)
	}
	initialized_count += 1
	return true
}

deinitialize :: proc() {
	if initialized_count <= 0 {
		return
	}
	initialized_count -= 1
	if initialized_count == 0 {
		enet.deinitialize()
	}
}

server_create :: proc(port: u16, max_peers: uint, channel_count := DEFAULT_CHANNEL_COUNT) -> (host: Host, ok: bool) {
	if !initialize() {
		return {}, false
	}

	address := enet.Address{host = enet.HOST_ANY, port = port}
	handle := enet.host_create(&address, max_peers, channel_count, 0, 0)
	if handle == nil {
		deinitialize()
		return {}, false
	}

	return Host{handle = handle, mode = .Server}, true
}

client_create :: proc(channel_count := DEFAULT_CHANNEL_COUNT) -> (host: Host, ok: bool) {
	if !initialize() {
		return {}, false
	}

	handle := enet.host_create(nil, 1, channel_count, 0, 0)
	if handle == nil {
		deinitialize()
		return {}, false
	}

	return Host{handle = handle, mode = .Client}, true
}

destroy :: proc(host: ^Host) {
	if host.handle != nil {
		enet.host_destroy(host.handle)
		deinitialize()
	}
	host^ = {}
}

connect :: proc(host: ^Host, address_text: string, port: u16, channel_count := DEFAULT_CHANNEL_COUNT) -> (peer: Peer, ok: bool) {
	if host.handle == nil {
		return {}, false
	}

	address := enet.Address{port = port}
	address_c := strings.clone_to_cstring(address_text, context.temp_allocator)
	if enet.address_set_host(&address, address_c) != 0 {
		return {}, false
	}

	raw_peer := enet.host_connect(host.handle, &address, channel_count, 0)
	if raw_peer == nil {
		return {}, false
	}

	return peer_from_raw(raw_peer), true
}

poll :: proc(host: ^Host, receive_buffer: []byte, timeout_ms: u32 = 0) -> Event {
	if host.handle == nil {
		return {}
	}

	raw_event: enet.Event
	result := enet.host_service(host.handle, &raw_event, timeout_ms)
	if result <= 0 {
		return {}
	}

	#partial switch raw_event.type {
	case .CONNECT:
		return Event{kind = .Connect, peer = peer_from_raw(raw_event.peer), user_data = raw_event.data}
	case .DISCONNECT:
		return Event{kind = .Disconnect, peer = peer_from_raw(raw_event.peer), user_data = raw_event.data}
	case .RECEIVE:
		packet := raw_event.packet
		defer enet.packet_destroy(packet)
		data_len := int(packet.dataLength)
		copy_len := min(data_len, len(receive_buffer))
		if copy_len > 0 {
			mem.copy(&receive_buffer[0], packet.data, copy_len)
		}
		return Event{
			kind = .Receive,
			peer = peer_from_raw(raw_event.peer),
			channel = raw_event.channelID,
			data = receive_buffer[:copy_len],
			truncated = copy_len < data_len,
			user_data = raw_event.data,
		}
	}

	return {}
}

send :: proc(peer: Peer, channel: u8, data: []byte, mode: Packet_Mode) -> bool {
	raw_peer := peer_to_raw(peer)
	if raw_peer == nil {
		return false
	}

	packet := packet_create(data, mode)
	if packet == nil {
		return false
	}

	if enet.peer_send(raw_peer, channel, packet) != 0 {
		enet.packet_destroy(packet)
		return false
	}
	return true
}

broadcast :: proc(host: ^Host, channel: u8, data: []byte, mode: Packet_Mode) -> bool {
	if host.handle == nil {
		return false
	}

	packet := packet_create(data, mode)
	if packet == nil {
		return false
	}
	enet.host_broadcast(host.handle, channel, packet)
	return true
}

flush :: proc(host: ^Host) {
	if host.handle != nil {
		enet.host_flush(host.handle)
	}
}

disconnect :: proc(peer: Peer, data: u32 = 0) {
	if raw_peer := peer_to_raw(peer); raw_peer != nil {
		enet.peer_disconnect(raw_peer, data)
	}
}

disconnect_now :: proc(peer: Peer, data: u32 = 0) {
	if raw_peer := peer_to_raw(peer); raw_peer != nil {
		enet.peer_disconnect_now(raw_peer, data)
	}
}

peer_round_trip_ms :: proc(peer: Peer) -> u32 {
	if raw_peer := peer_to_raw(peer); raw_peer != nil {
		return raw_peer.roundTripTime
	}
	return 0
}

packet_create :: proc(data: []byte, mode: Packet_Mode) -> ^enet.Packet {
	flags := enet.PacketFlags{}
	if mode == .Reliable {
		flags += {.RELIABLE}
	}

	data_ptr: rawptr
	if len(data) > 0 {
		data_ptr = raw_data(data)
	}
	return enet.packet_create(data_ptr, uint(len(data)), flags)
}

peer_from_raw :: proc(peer: ^enet.Peer) -> Peer {
	return Peer(uintptr(peer))
}

peer_to_raw :: proc(peer: Peer) -> ^enet.Peer {
	return cast(^enet.Peer)uintptr(peer)
}

package main

import protocol "../protocol"
import transport "../net"

NET_IMPAIRMENT_QUEUE_CAPACITY :: 512

NetImpairment :: struct {
	enabled: bool,
	delay_ms: f32,
	jitter_ms: f32,
	loss_percent: f32,
	duplicate_percent: f32,
	reorder_percent: f32,
	rng_state: u32,
	queued: [NET_IMPAIRMENT_QUEUE_CAPACITY]NetImpairedPacket,
	queued_count: int,
	dropped_queue_full: u32,
}

NetImpairedPacket :: struct {
	due_time: f64,
	peer: transport.Peer,
	channel: u8,
	mode: transport.Packet_Mode,
	length: int,
	data: [protocol.MAX_PACKET_SIZE]byte,
}

net_impairment_enabled :: proc(options: Options) -> bool {
	return options.net_down_delay_ms > 0 || options.net_down_jitter_ms > 0 || options.net_down_loss > 0 || options.net_down_duplicate > 0 || options.net_down_reorder > 0
}

net_impairment_init :: proc(impairment: ^NetImpairment, options: Options) {
	seed := options.net_seed
	if seed == 0 {
		seed = 0x12345678
	}
	impairment^ = {
		enabled = net_impairment_enabled(options),
		delay_ms = options.net_down_delay_ms,
		jitter_ms = options.net_down_jitter_ms,
		loss_percent = clamp_percent(options.net_down_loss),
		duplicate_percent = clamp_percent(options.net_down_duplicate),
		reorder_percent = clamp_percent(options.net_down_reorder),
		rng_state = seed,
	}
}

net_impairment_queue_packet :: proc(impairment: ^NetImpairment, peer: transport.Peer, channel: u8, data: []byte, mode: transport.Packet_Mode, now_seconds: f64) -> bool {
	if !impairment.enabled {
		return false
	}
	if net_impairment_roll(impairment, impairment.loss_percent) {
		return true
	}
	due_time := net_impairment_due_time(impairment, now_seconds)
	if net_impairment_roll(impairment, impairment.reorder_percent) {
		due_time += f64(max(impairment.delay_ms + impairment.jitter_ms, f32(1))) / 1000.0
	}
	net_impairment_insert(impairment, peer, channel, data, mode, due_time)
	if net_impairment_roll(impairment, impairment.duplicate_percent) {
		duplicate_due_time := net_impairment_due_time(impairment, now_seconds)
		net_impairment_insert(impairment, peer, channel, data, mode, duplicate_due_time)
	}
	return true
}

net_impairment_due_time :: proc(impairment: ^NetImpairment, now_seconds: f64) -> f64 {
	delay_ms := impairment.delay_ms
	if impairment.jitter_ms > 0 {
		delay_ms += (net_impairment_random_unit(impairment) * 2 - 1) * impairment.jitter_ms
	}
	if delay_ms < 0 {
		delay_ms = 0
	}
	return now_seconds + f64(delay_ms) / 1000.0
}

net_impairment_insert :: proc(impairment: ^NetImpairment, peer: transport.Peer, channel: u8, data: []byte, mode: transport.Packet_Mode, due_time: f64) {
	if impairment.queued_count >= NET_IMPAIRMENT_QUEUE_CAPACITY {
		copy(impairment.queued[:], impairment.queued[1:])
		impairment.queued_count -= 1
		impairment.dropped_queue_full += 1
	}
	insert_at := impairment.queued_count
	for insert_at > 0 && impairment.queued[insert_at - 1].due_time > due_time {
		impairment.queued[insert_at] = impairment.queued[insert_at - 1]
		insert_at -= 1
	}
	packet := NetImpairedPacket{due_time = due_time, peer = peer, channel = channel, mode = mode, length = len(data)}
	copy(packet.data[:], data[:packet.length])
	impairment.queued[insert_at] = packet
	impairment.queued_count += 1
}

net_impairment_pop_due :: proc(impairment: ^NetImpairment, now_seconds: f64) -> (packet: NetImpairedPacket, ok: bool) {
	if !impairment.enabled || impairment.queued_count == 0 || impairment.queued[0].due_time > now_seconds {
		return {}, false
	}
	packet = impairment.queued[0]
	if impairment.queued_count > 1 {
		copy(impairment.queued[:], impairment.queued[1:impairment.queued_count])
	}
	impairment.queued_count -= 1
	return packet, true
}

net_impairment_drop_peer :: proc(impairment: ^NetImpairment, peer: transport.Peer) {
	write_index := 0
	for read_index in 0..<impairment.queued_count {
		if impairment.queued[read_index].peer == peer {
			continue
		}
		if write_index != read_index {
			impairment.queued[write_index] = impairment.queued[read_index]
		}
		write_index += 1
	}
	impairment.queued_count = write_index
}

net_impairment_roll :: proc(impairment: ^NetImpairment, percent: f32) -> bool {
	if percent <= 0 {
		return false
	}
	return net_impairment_random_unit(impairment) * 100 < percent
}

net_impairment_random_unit :: proc(impairment: ^NetImpairment) -> f32 {
	state := impairment.rng_state
	state = state ~ (state << 13)
	state = state ~ (state >> 17)
	state = state ~ (state << 5)
	if state == 0 {
		state = 0x12345678
	}
	impairment.rng_state = state
	return f32(state & 0x00ffffff) / f32(0x01000000)
}

clamp_percent :: proc(value: f32) -> f32 {
	return min(max(value, 0), 100)
}

package main

import "core:testing"
import protocol "../protocol"
import transport "../net"

@(test)
test_net_impairment_queues_packet_until_due_time :: proc(t: ^testing.T) {
	impairment := test_net_impairment({net_down_delay_ms = 100, net_seed = 1})
	defer free(impairment)
	data := []byte{1, 2, 3}

	queued := net_impairment_queue_packet(impairment, 4, protocol.CHANNEL_SNAPSHOTS, data, .Unreliable, 10)
	early, early_ok := net_impairment_pop_due(impairment, 10.05)
	_ = early
	due, due_ok := net_impairment_pop_due(impairment, 10.1)

	testing.expect(t, queued, "enabled impairment should consume snapshot packet")
	testing.expect(t, !early_ok, "packet should remain queued before due time")
	testing.expect(t, due_ok, "packet should pop at due time")
	testing.expect_value(t, due.peer, transport.Peer(4))
	testing.expect_value(t, due.channel, protocol.CHANNEL_SNAPSHOTS)
	testing.expect_value(t, due.length, 3)
	testing.expect_value(t, due.data[0], byte(1))
}

@(test)
test_net_impairment_orders_packets_by_due_time :: proc(t: ^testing.T) {
	impairment := test_net_impairment({net_seed = 1})
	defer free(impairment)
	net_impairment_insert(impairment, 1, protocol.CHANNEL_SNAPSHOTS, []byte{3}, .Unreliable, 3)
	net_impairment_insert(impairment, 1, protocol.CHANNEL_SNAPSHOTS, []byte{1}, .Unreliable, 1)
	net_impairment_insert(impairment, 1, protocol.CHANNEL_SNAPSHOTS, []byte{2}, .Unreliable, 2)

	first, first_ok := net_impairment_pop_due(impairment, 3)
	second, second_ok := net_impairment_pop_due(impairment, 3)
	third, third_ok := net_impairment_pop_due(impairment, 3)

	testing.expect(t, first_ok && second_ok && third_ok, "all inserted packets should be due")
	testing.expect_value(t, first.data[0], byte(1))
	testing.expect_value(t, second.data[0], byte(2))
	testing.expect_value(t, third.data[0], byte(3))
}

@(test)
test_net_impairment_loss_consumes_without_queueing :: proc(t: ^testing.T) {
	impairment := test_net_impairment({net_down_loss = 100, net_seed = 1})
	defer free(impairment)

	queued := net_impairment_queue_packet(impairment, 1, protocol.CHANNEL_SNAPSHOTS, []byte{1}, .Unreliable, 0)

	testing.expect(t, queued, "lost packet should be consumed by impairment")
	testing.expect_value(t, impairment.queued_count, 0)
}

@(test)
test_net_impairment_duplicate_adds_second_packet :: proc(t: ^testing.T) {
	impairment := test_net_impairment({net_down_duplicate = 100, net_seed = 1})
	defer free(impairment)

	net_impairment_queue_packet(impairment, 1, protocol.CHANNEL_SNAPSHOTS, []byte{9}, .Unreliable, 0)

	testing.expect_value(t, impairment.queued_count, 2)
}

@(test)
test_net_impairment_queue_overflow_tracks_drops :: proc(t: ^testing.T) {
	impairment := test_net_impairment({net_seed = 1})
	defer free(impairment)
	for i in 0..<NET_IMPAIRMENT_QUEUE_CAPACITY + 1 {
		net_impairment_insert(impairment, 1, protocol.CHANNEL_SNAPSHOTS, []byte{byte(i % 255)}, .Unreliable, f64(i))
	}

	testing.expect_value(t, impairment.queued_count, NET_IMPAIRMENT_QUEUE_CAPACITY)
	testing.expect_value(t, impairment.dropped_queue_full, u32(1))
	first, ok := net_impairment_pop_due(impairment, f64(NET_IMPAIRMENT_QUEUE_CAPACITY + 1))
	testing.expect(t, ok, "overflow should keep later queued packets")
	testing.expect_value(t, first.due_time, f64(1))
}

@(test)
test_net_impairment_drop_peer_removes_queued_packets :: proc(t: ^testing.T) {
	impairment := test_net_impairment({net_seed = 1})
	defer free(impairment)
	net_impairment_insert(impairment, 1, protocol.CHANNEL_SNAPSHOTS, []byte{1}, .Unreliable, 1)
	net_impairment_insert(impairment, 2, protocol.CHANNEL_SNAPSHOTS, []byte{2}, .Unreliable, 2)
	net_impairment_insert(impairment, 1, protocol.CHANNEL_SNAPSHOTS, []byte{3}, .Unreliable, 3)

	net_impairment_drop_peer(impairment, 1)

	testing.expect_value(t, impairment.queued_count, 1)
	packet, ok := net_impairment_pop_due(impairment, 3)
	testing.expect(t, ok, "remaining peer packet should still be queued")
	testing.expect_value(t, packet.peer, transport.Peer(2))
	testing.expect_value(t, packet.data[0], byte(2))
}

test_net_impairment :: proc(options: Options) -> ^NetImpairment {
	impairment := new(NetImpairment)
	net_impairment_init(impairment, options)
	if !impairment.enabled {
		impairment.enabled = true
	}
	return impairment
}

#+test
package game

import "core:testing"
import engine "../engine"
import "core:math/linalg"
import protocol "../protocol"

@(test)
test_net_client_recent_user_cmds_sends_latest_unacked :: proc(t: ^testing.T) {
	net: GameNetClient
	for sequence: u32 = 1; sequence <= 10; sequence += 1 {
		net.command_sequence = sequence
		game_net_client_store_user_cmd(&net, test_net_user_cmd(sequence))
	}

	cmds := game_net_client_recent_user_cmds(&net)
	testing.expect_value(t, cmds.count, u8(protocol.MAX_USER_CMDS_PER_PACKET))
	testing.expect_value(t, cmds.cmds[0].sequence, u32(3))
	testing.expect_value(t, cmds.cmds[int(cmds.count) - 1].sequence, u32(10))
}

@(test)
test_net_client_recent_user_cmds_starts_after_ack :: proc(t: ^testing.T) {
	net: GameNetClient
	for sequence: u32 = 1; sequence <= 10; sequence += 1 {
		net.command_sequence = sequence
		game_net_client_store_user_cmd(&net, test_net_user_cmd(sequence))
	}
	net.last_server_acked_command = 6

	cmds := game_net_client_recent_user_cmds(&net)
	testing.expect_value(t, cmds.count, u8(4))
	testing.expect_value(t, cmds.cmds[0].sequence, u32(7))
	testing.expect_value(t, cmds.cmds[3].sequence, u32(10))
}

@(test)
test_net_client_recent_user_cmds_clamps_to_available_history :: proc(t: ^testing.T) {
	net: GameNetClient
	for sequence: u32 = 1; sequence <= CLIENT_COMMAND_HISTORY + 5; sequence += 1 {
		net.command_sequence = sequence
		game_net_client_store_user_cmd(&net, test_net_user_cmd(sequence))
	}
	net.last_server_acked_command = 1

	cmds := game_net_client_recent_user_cmds(&net)
	testing.expect_value(t, cmds.count, u8(protocol.MAX_USER_CMDS_PER_PACKET))
	testing.expect_value(t, cmds.cmds[0].sequence, u32(CLIENT_COMMAND_HISTORY - 2))
	testing.expect_value(t, cmds.cmds[int(cmds.count) - 1].sequence, u32(CLIENT_COMMAND_HISTORY + 5))
}

@(test)
test_net_client_apply_snapshot_ack_is_monotonic :: proc(t: ^testing.T) {
	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID, last_server_acked_command = 10}
	scene: Scene

	game_net_client_apply_snapshot(&net, &scene, protocol.Server_Snapshot{sequence = 1, last_processed_user_cmd = 9})
	testing.expect_value(t, net.last_server_acked_command, u32(10))

	game_net_client_apply_snapshot(&net, &scene, protocol.Server_Snapshot{sequence = 2, last_processed_user_cmd = 12})
	testing.expect_value(t, net.last_server_acked_command, u32(12))
}

@(test)
test_net_client_apply_snapshot_rejects_stale_sequence :: proc(t: ^testing.T) {
	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID}
	scene := test_net_scene()
	defer delete(scene.players)

	newer := protocol.Server_Snapshot{sequence = 2, server_tick = 10, player_count = 1}
	newer.players[0] = {player_id = 2, position = {10, 0, 0}, yaw = 1}
	game_net_client_apply_snapshot(&net, &scene, newer)

	stale := protocol.Server_Snapshot{sequence = 1, server_tick = 11, player_count = 1}
	stale.players[0] = {player_id = 2, position = {99, 0, 0}, yaw = 9}
	game_net_client_apply_snapshot(&net, &scene, stale)

	player := scene_player_record(&scene, 2)
	testing.expect(t, player != nil, "newer snapshot should create remote player")
	testing.expect_value(t, player.remote_sample_count, 1)
	testing.expect_value(t, player.remote_samples[0].server_tick, u32(10))
	testing.expect_value(t, player.remote_samples[0].position, Vec3{10, 0, 0})
}

@(test)
test_net_client_apply_snapshot_accepts_out_of_order_clusters_from_same_sequence :: proc(t: ^testing.T) {
	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID}
	scene := test_net_scene()
	defer delete(scene.players)
	defer delete(scene.objects)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)

	prop_cluster := protocol.Server_Snapshot{sequence = 3, cluster_index = 1, server_tick = 20, prop_count = 1}
	prop_cluster.props[0] = {net_id = 55, prop_asset_index = 0, position = {5, 0, 0}, rotation = {0, 0, 0, 1}}
	game_net_client_apply_snapshot(&net, &scene, prop_cluster)

	player_cluster := protocol.Server_Snapshot{sequence = 3, cluster_index = 0, server_tick = 20, player_count = 1, last_processed_user_cmd = 7}
	player_cluster.players[0] = {player_id = 2, position = {2, 0, 0}, ground_normal = {0, 1, 0}}
	game_net_client_apply_snapshot(&net, &scene, player_cluster)

	testing.expect(t, scene_object_by_net_id(&scene, 55) != nil, "out-of-order prop cluster should apply")
	player := scene_player_record(&scene, 2)
	testing.expect(t, player != nil, "later-arriving player cluster from same sequence should not be dropped as stale")
	testing.expect_value(t, net.last_server_acked_command, u32(7))
}

@(test)
test_net_client_apply_snapshot_rejects_duplicate_cluster :: proc(t: ^testing.T) {
	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID}
	scene := test_net_scene()
	defer delete(scene.players)

	first := protocol.Server_Snapshot{sequence = 4, cluster_index = 0, server_tick = 10, player_count = 1}
	first.players[0] = {player_id = 2, position = {1, 0, 0}}
	game_net_client_apply_snapshot(&net, &scene, first)

	duplicate := protocol.Server_Snapshot{sequence = 4, cluster_index = 0, server_tick = 10, player_count = 1}
	duplicate.players[0] = {player_id = 2, position = {9, 0, 0}}
	game_net_client_apply_snapshot(&net, &scene, duplicate)

	player := scene_player_record(&scene, 2)
	testing.expect(t, player != nil, "first cluster should create remote player")
	testing.expect_value(t, player.remote_samples[0].position, Vec3{1, 0, 0})
}

@(test)
test_net_client_loopback_applies_local_authoritative_snapshot :: proc(t: ^testing.T) {
	local_server := new(NetServer)
	defer free(local_server)
	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID, local_server = local_server}
	scene := test_net_scene()
	defer delete(scene.players)
	scene_add_player(&scene, LOCAL_PLAYER_ID, {0, 0, 0}, 0)

	snapshot := protocol.Server_Snapshot{sequence = 1, server_tick = 8, player_count = 1}
	snapshot.players[0] = {
		player_id = LOCAL_PLAYER_ID,
		position = {4, 2, -3},
		velocity = {1, -2, 3},
		yaw = 1.25,
		pitch = -0.5,
		grounded = true,
		ground_normal = {0, 1, 0},
	}
	game_net_client_apply_snapshot(&net, &scene, snapshot)

	player := scene_player(&scene, LOCAL_PLAYER_ID)
	testing.expect(t, player != nil, "local player should exist")
	testing.expect_value(t, player.position, Vec3{4, 2, -3})
	testing.expect_value(t, player.velocity, Vec3{1, -2, 3})
	testing.expect_value(t, player.yaw, f32(1.25))
	testing.expect_value(t, player.pitch, f32(-0.5))
	testing.expect_value(t, player.grounded, true)
	testing.expect_value(t, player.ground_normal, Vec3{0, 1, 0})
}

@(test)
test_net_client_reconcile_records_prediction_error :: proc(t: ^testing.T) {
	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID, command_sequence = 1}
	net.prediction_history[1 % CLIENT_COMMAND_HISTORY] = PredictedPlayerState{
		sequence = 1,
		position = {3, 0, 0},
	}
	scene := test_net_scene()
	defer delete(scene.players)
	scene_add_player(&scene, LOCAL_PLAYER_ID, {0, 0, 0}, 0)

	snapshot := protocol.Server_Snapshot{sequence = 1, last_processed_user_cmd = 1, player_count = 1}
	snapshot.players[0] = {
		player_id = LOCAL_PLAYER_ID,
		position = {1, 0, 0},
		ground_normal = {0, 1, 0},
	}
	game_net_client_apply_snapshot(&net, &scene, snapshot)

	testing.expect_value(t, net.last_prediction_error, f32(2))
	testing.expect_value(t, net.prediction_correction_count, u32(1))
	testing.expect_value(t, net.last_prediction_replay_count, u32(0))
}

@(test)
test_net_client_reconcile_replays_unacked_command :: proc(t: ^testing.T) {
	physics_test_lock()
	defer physics_test_unlock()

	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID, command_sequence = 2}
	net.command_history[2 % CLIENT_COMMAND_HISTORY] = test_net_user_cmd(2)
	scene := test_net_physics_scene()
	defer test_net_physics_scene_destroy(&scene)
	scene_add_player(&scene, LOCAL_PLAYER_ID, {0, 1, 0}, 0)

	snapshot := protocol.Server_Snapshot{sequence = 1, last_processed_user_cmd = 1, player_count = 1}
	snapshot.players[0] = {
		player_id = LOCAL_PLAYER_ID,
		position = {0, 1, 0},
		ground_normal = {0, 1, 0},
	}
	game_net_client_apply_snapshot(&net, &scene, snapshot)

	player := scene_player(&scene, LOCAL_PLAYER_ID)
	testing.expect(t, player != nil, "local player should exist")
	testing.expect_value(t, net.last_prediction_replay_count, u32(1))
	testing.expect(t, player.position != Vec3{0, 1, 0}, "unacked command should be replayed from authoritative state")
	_, ok := game_net_client_predicted_state(&net, 2)
	testing.expect(t, ok, "replayed command should refresh prediction history")
}

@(test)
test_net_client_apply_snapshot_upserts_replicated_prop :: proc(t: ^testing.T) {
	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID}
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)

	snapshot := protocol.Server_Snapshot{sequence = 1, server_tick = 10, prop_count = 1}
	snapshot.props[0] = {
		net_id = 12,
		prop_asset_index = 0,
		position = {2, 3, 4},
		rotation = {0, 0.5, 0, 0.8660254},
	}
	game_net_client_apply_snapshot(&net, &scene, snapshot)

	object := scene_object_by_net_id(&scene, 12)
	testing.expect(t, object != nil, "replicated prop should be created")
	testing.expect_value(t, object.kind, ObjectKind.Prop)
	testing.expect_value(t, object.transform.position, Vec3{2, 3, 4})
	testing.expect_value(t, object.render_rotation.y, f32(0.5))
	testing.expect_value(t, object.replica.net_id, protocol.NetId(12))
	testing.expect_value(t, object.prop_asset_index, u16(0))
	testing.expect_value(t, object.replica.transform_buffer.count, 1)
	testing.expect_value(t, object.replica.transform_buffer.samples[0].server_tick, u32(10))
	testing.expect(t, !object.physics.enabled, "replicated client prop should be render-only")
}

@(test)
test_net_client_apply_snapshot_updates_existing_prop_as_render_only :: proc(t: ^testing.T) {
	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID}
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)
	append(&scene.objects, Object{
		id = 4,
		name = "Prop",
		kind = .Prop,
		replica = {net_id = 4, kind = .Prop, authority = .ServerAuthoritative},
		transform = {position = {0, 0, 0}},
		render_rotation = linalg.QUATERNIONF32_IDENTITY,
		render = {visible = true},
		physics = {enabled = true, sync_transform = true},
	})

	snapshot := protocol.Server_Snapshot{sequence = 1, server_tick = 8, prop_count = 1}
	snapshot.props[0] = {
		net_id = 4,
		prop_asset_index = 0,
		position = {9, 8, 7},
		rotation = {0.1, 0.2, 0.3, 0.9},
	}
	game_net_client_apply_snapshot(&net, &scene, snapshot)

	object := scene_object(&scene, ObjectId(4))
	testing.expect(t, object != nil, "existing prop should remain")
	testing.expect_value(t, object.transform.position, Vec3{9, 8, 7})
	testing.expect_value(t, object.render_rotation.z, f32(0.3))
	testing.expect_value(t, object.replica.transform_buffer.count, 1)
	testing.expect(t, !object.physics.enabled && !object.physics.sync_transform, "snapshot should take over local prop physics")
}

@(test)
test_net_client_apply_snapshot_removes_replicated_prop :: proc(t: ^testing.T) {
	net := GameNetClient{local_player_id = LOCAL_PLAYER_ID}
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)
	append(&scene.objects, Object{id = 7, name = "Prop", kind = .Prop, render = {visible = true}, replica = {net_id = 7, kind = .Prop, authority = .ServerAuthoritative}})

	snapshot := protocol.Server_Snapshot{sequence = 1, removed_prop_count = 1}
	snapshot.removed_prop_ids[0] = 7
	game_net_client_apply_snapshot(&net, &scene, snapshot)

	testing.expect(t, scene_object_by_net_id(&scene, 7) == nil, "removed prop id should delete local replica")
}

@(test)
test_scene_object_replicated_prop_interpolates_samples :: proc(t: ^testing.T) {
	buffer: ReplicatedTransformBuffer
	rot_a := linalg.QUATERNIONF32_IDENTITY
	rot_b := linalg.quaternion_from_euler_angle_y_f32(1)
	replicated_transform_add_sample(&buffer, {server_tick = 10, position = {0, 0, 0}, rotation = rot_a})
	replicated_transform_add_sample(&buffer, {server_tick = 14, position = {4, 0, 0}, rotation = rot_b})

	position, _ := replicated_transform_at_tick(&buffer, {}, linalg.QUATERNIONF32_IDENTITY, 12)
	testing.expect_value(t, position, Vec3{2, 0, 0})
}

@(test)
test_scene_object_replicated_prop_collision_sample_extrapolates_from_velocity :: proc(t: ^testing.T) {
	buffer: ReplicatedTransformBuffer
	replicated_transform_add_sample(&buffer, {server_tick = 10, position = {1, 0, 0}, rotation = linalg.QUATERNIONF32_IDENTITY, linear_velocity = {64, 0, 0}})

	position, _ := replicated_collision_transform_at_tick(&buffer, {}, linalg.QUATERNIONF32_IDENTITY, 12)
	testing.expect_value(t, position, Vec3{3, 0, 0})
}

@(test)
test_net_server_prop_delta_skips_unchanged_sent_prop_until_refresh :: proc(t: ^testing.T) {
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)
	append(&scene.objects, Object{id = 3, kind = .Prop, transform = {position = {1, 2, 3}}, render_rotation = linalg.QUATERNIONF32_IDENTITY, replica = {net_id = 3, kind = .Prop, authority = .ServerAuthoritative}})
	server := new(NetServer)
	defer free(server)
	server^ = {scene = &scene, server_tick = 10}
	session: NetServerSession
	snapshot: protocol.Server_Snapshot

	net_server_add_prop_delta(server, &session, &snapshot)
	testing.expect_value(t, snapshot.prop_count, u16(1))

	snapshot = {}
	net_server_add_prop_delta(server, &session, &snapshot)
	testing.expect_value(t, snapshot.prop_count, u16(0))

	server.server_tick += NET_SERVER_PROP_REFRESH_TICKS
	net_server_add_prop_delta(server, &session, &snapshot)
	testing.expect_value(t, snapshot.prop_count, u16(1))
}

@(test)
test_net_server_prop_delta_sends_removed_sent_prop :: proc(t: ^testing.T) {
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)
	server := new(NetServer)
	defer free(server)
	server^ = {scene = &scene, server_tick = 1}
	session: NetServerSession
	session.sent_props[0] = {active = true, net_id = 44}
	snapshot: protocol.Server_Snapshot

	net_server_add_prop_delta(server, &session, &snapshot)
	testing.expect_value(t, snapshot.removed_prop_count, u16(1))
	testing.expect_value(t, snapshot.removed_prop_ids[0], protocol.NetId(44))
	testing.expect(t, session.sent_props[0].active, "removed prop should keep sent state as a tombstone")
	testing.expect(t, session.sent_props[0].pending_removal, "removed prop should remain a pending tombstone")
}

@(test)
test_net_server_prop_delta_keeps_removed_tombstones_when_packet_budget_fills :: proc(t: ^testing.T) {
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)
	server := new(NetServer)
	defer free(server)
	server^ = {scene = &scene, server_tick = 1}
	session: NetServerSession
	for i in 0..<protocol.MAX_REMOVED_PROPS + 1 {
		session.sent_props[i] = {active = true, net_id = protocol.NetId(u32(i + 1))}
	}
	snapshot: protocol.Server_Snapshot

	more_pending := net_server_add_prop_delta(server, &session, &snapshot)

	testing.expect(t, more_pending, "one removal should remain pending after the per-packet removal array fills")
	testing.expect_value(t, snapshot.removed_prop_count, u16(protocol.MAX_REMOVED_PROPS))
	testing.expect(t, session.sent_props[protocol.MAX_REMOVED_PROPS].pending_removal, "overflow removal should stay pending for a later cluster")
}

@(test)
test_net_server_prop_delta_skips_client_only_prop :: proc(t: ^testing.T) {
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)
	append(&scene.objects, Object{id = 5, kind = .Prop, replica = {net_id = 5, kind = .Prop, authority = .ClientOnly}})
	server := new(NetServer)
	defer free(server)
	server^ = {scene = &scene}
	session: NetServerSession
	snapshot: protocol.Server_Snapshot

	net_server_add_prop_delta(server, &session, &snapshot)
	testing.expect_value(t, snapshot.prop_count, u16(0))
}

@(test)
test_net_server_prop_delta_round_robins_more_than_one_snapshot_of_props :: proc(t: ^testing.T) {
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)
	PROP_COUNT :: protocol.MAX_SNAPSHOT_PROPS + 16
	for i in 0..<PROP_COUNT {
		net_id := protocol.NetId(u32(i + 1))
		append(&scene.objects, Object{id = ObjectId(u32(i + 1)), kind = .Prop, transform = {position = {f32(i), 0, 0}}, render_rotation = linalg.QUATERNIONF32_IDENTITY, replica = {net_id = net_id, kind = .Prop, authority = .ServerAuthoritative}})
	}
	server := new(NetServer)
	defer free(server)
	server^ = {scene = &scene, server_tick = 1}
	session: NetServerSession
	first_snapshot: protocol.Server_Snapshot
	second_snapshot: protocol.Server_Snapshot

	net_server_add_prop_delta(server, &session, &first_snapshot)
	net_server_add_prop_delta(server, &session, &second_snapshot)

	testing.expect_value(t, first_snapshot.prop_count, u16(protocol.MAX_SNAPSHOT_PROPS))
	testing.expect_value(t, second_snapshot.prop_count, u16(16))
	testing.expect_value(t, second_snapshot.props[0].net_id, protocol.NetId(protocol.MAX_SNAPSHOT_PROPS + 1))
	testing.expect(t, net_server_sent_prop(&session, protocol.NetId(PROP_COUNT)) != nil, "sent prop table should store props beyond one packet budget")
}

@(test)
test_net_server_prop_delta_continues_after_packet_budget_with_dirty_sent_props :: proc(t: ^testing.T) {
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)
	PROP_COUNT :: protocol.MAX_SNAPSHOT_PROPS + 8
	for i in 0..<PROP_COUNT {
		net_id := protocol.NetId(u32(i + 1))
		append(&scene.objects, Object{id = ObjectId(u32(i + 1)), kind = .Prop, transform = {position = {f32(i), 0, 0}}, render_rotation = linalg.QUATERNIONF32_IDENTITY, replica = {net_id = net_id, kind = .Prop, authority = .ServerAuthoritative}})
	}
	server := new(NetServer)
	defer free(server)
	server^ = {scene = &scene, server_tick = 1}
	session: NetServerSession
	snapshot: protocol.Server_Snapshot

	net_server_add_prop_delta(server, &session, &snapshot)
	for &object in scene.objects[:protocol.MAX_SNAPSHOT_PROPS] {
		object.transform.position.x += 1
	}
	server.server_tick += 1
	snapshot = {}
	net_server_add_prop_delta(server, &session, &snapshot)

	testing.expect_value(t, snapshot.prop_count, u16(protocol.MAX_SNAPSHOT_PROPS))
	testing.expect_value(t, snapshot.props[0].net_id, protocol.NetId(protocol.MAX_SNAPSHOT_PROPS + 1))
	testing.expect_value(t, snapshot.props[7].net_id, protocol.NetId(PROP_COUNT))
}

@(test)
test_net_server_snapshot_prop_budget_targets_mtu_sized_packets :: proc(t: ^testing.T) {
	snapshot := protocol.Server_Snapshot{player_count = 1}
	budget := 0
	for protocol.server_snapshot_can_add_prop(snapshot, NET_SERVER_SNAPSHOT_TARGET_BYTES) {
		snapshot.prop_count += 1
		budget += 1
	}
	packet_bytes := protocol.server_snapshot_packet_size(snapshot)

	testing.expect(t, budget > 0, "snapshot should reserve some prop budget")
	testing.expect(t, packet_bytes <= NET_SERVER_SNAPSHOT_TARGET_BYTES, "snapshot prop budget should stay within target bytes")
}

@(test)
test_net_server_broadcast_snapshot_splits_many_props_into_clusters :: proc(t: ^testing.T) {
	scene := test_net_scene()
	defer delete(scene.objects)
	defer delete(scene.players)
	scene.objects = make([dynamic]Object, 0, MAX_OBJECTS)
	PROP_COUNT :: 80
	for i in 0..<PROP_COUNT {
		net_id := protocol.NetId(u32(i + 1))
		append(&scene.objects, Object{id = ObjectId(u32(i + 1)), kind = .Prop, transform = {position = {f32(i), 0, 0}}, render_rotation = linalg.QUATERNIONF32_IDENTITY, replica = {net_id = net_id, kind = .Prop, authority = .ServerAuthoritative}})
	}
	server := new(NetServer)
	defer free(server)
	server^ = {scene = &scene, server_tick = 2}
	server.sessions[0] = {peer = 1, active = true, accepted = true, player_id = 1}

	net_server_broadcast_snapshot(server)

	total_props := 0
	packet_count := 0
	for {
		packet, ok := net_server_poll_outgoing(server)
		if !ok {
			break
		}
		parsed, err := protocol.parse_packet(packet.data[:packet.length], packet.channel)
		testing.expect_value(t, err, protocol.Parse_Error.None)
		testing.expect(t, int(packet.length) <= NET_SERVER_SNAPSHOT_TARGET_BYTES, "cluster should stay within target bytes")
		total_props += int(parsed.snapshot.prop_count)
		packet_count += 1
	}

	testing.expect(t, packet_count > 1, "many props should split across more than one snapshot cluster")
	testing.expect_value(t, total_props, PROP_COUNT)
	testing.expect_value(t, server.snapshot_stats.packet_count, packet_count)
	testing.expect_value(t, server.snapshot_stats.prop_count, PROP_COUNT)
	testing.expect_value(t, server.snapshot_stats.cluster_limit_deferred_sessions, 0)
}

@(test)
test_scene_player_interpolates_remote_samples :: proc(t: ^testing.T) {
	scene := test_net_scene()
	defer delete(scene.players)
	player := scene_add_player_record(&scene, 2, {0, 0, 0}, 0)
	scene_player_add_remote_sample(player, {server_tick = 10, position = {0, 0, 0}, yaw = 0})
	scene_player_add_remote_sample(player, {server_tick = 14, position = {4, 0, 0}, yaw = 4})

	transform := scene_player_interpolated_transform(player, 12)
	testing.expect_value(t, transform.position, Vec3{2, 0, 0})
	testing.expect_value(t, transform.yaw, f32(2))
}

@(test)
test_scene_player_remote_samples_sort_and_update_duplicates :: proc(t: ^testing.T) {
	scene := test_net_scene()
	defer delete(scene.players)
	player := scene_add_player_record(&scene, 2, {0, 0, 0}, 0)
	scene_player_add_remote_sample(player, {server_tick = 12, position = {12, 0, 0}, yaw = 12})
	scene_player_add_remote_sample(player, {server_tick = 10, position = {10, 0, 0}, yaw = 10})
	scene_player_add_remote_sample(player, {server_tick = 12, position = {99, 0, 0}, yaw = 99})

	testing.expect_value(t, player.remote_sample_count, 2)
	testing.expect_value(t, player.remote_samples[0].server_tick, u32(10))
	testing.expect_value(t, player.remote_samples[1].server_tick, u32(12))
	testing.expect_value(t, player.remote_samples[1].position, Vec3{99, 0, 0})
}

test_net_user_cmd :: proc(sequence: u32) -> protocol.User_Cmd {
	return {
		sequence = sequence,
		client_tick = sequence,
		move_forward = f32(sequence),
		yaw = f32(sequence) * 0.1,
	}
}

test_net_scene :: proc() -> Scene {
	return Scene{
		players = make([dynamic]ScenePlayer, 0, MAX_PLAYERS),
		camera_player_id = LOCAL_PLAYER_ID,
	}
}

test_net_physics_scene :: proc() -> Scene {
	return Scene{
		physics = engine.physics_create(),
		players = make([dynamic]ScenePlayer, 0, MAX_PLAYERS),
		camera_player_id = LOCAL_PLAYER_ID,
	}
}

test_net_physics_scene_destroy :: proc(scene: ^Scene) {
	engine.physics_destroy(&scene.physics)
	delete(scene.players)
}

package game

import "core:testing"
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

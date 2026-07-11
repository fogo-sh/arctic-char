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

	game_net_client_apply_snapshot(&net, &scene, protocol.Server_Snapshot{last_processed_user_cmd = 9})
	testing.expect_value(t, net.last_server_acked_command, u32(10))

	game_net_client_apply_snapshot(&net, &scene, protocol.Server_Snapshot{last_processed_user_cmd = 12})
	testing.expect_value(t, net.last_server_acked_command, u32(12))
}

test_net_user_cmd :: proc(sequence: u32) -> protocol.User_Cmd {
	return {
		sequence = sequence,
		client_tick = sequence,
		move_forward = f32(sequence),
		yaw = f32(sequence) * 0.1,
	}
}

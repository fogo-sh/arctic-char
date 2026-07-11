package main

import "core:testing"
import protocol "../protocol"

@(test)
test_server_command_queue_consumes_in_sequence_order :: proc(t: ^testing.T) {
	session: ClientSession
	server_enqueue_user_cmd(&session, test_server_user_cmd(3))
	server_enqueue_user_cmd(&session, test_server_user_cmd(1))
	server_enqueue_user_cmd(&session, test_server_user_cmd(2))

	testing.expect_value(t, session.pending_cmd_count, 3)
	testing.expect_value(t, session.pending_cmds[0].sequence, u32(1))
	testing.expect_value(t, session.pending_cmds[1].sequence, u32(2))
	testing.expect_value(t, session.pending_cmds[2].sequence, u32(3))

	cmd, ok := server_consume_user_cmds(&session)
	testing.expect(t, ok, "queued commands should consume")
	testing.expect_value(t, cmd.sequence, u32(3))
	testing.expect_value(t, session.last_processed_cmd_sequence, u32(3))
	testing.expect_value(t, session.pending_cmd_count, 0)
	testing.expect(t, session.has_last_input_cmd, "consumed command should become fallback input")
}

@(test)
test_server_command_queue_ignores_duplicates_and_stale :: proc(t: ^testing.T) {
	session := ClientSession{last_processed_cmd_sequence = 5}
	server_enqueue_user_cmd(&session, test_server_user_cmd(4))
	server_enqueue_user_cmd(&session, test_server_user_cmd(5))
	server_enqueue_user_cmd(&session, test_server_user_cmd(6))
	server_enqueue_user_cmd(&session, test_server_user_cmd(6))

	testing.expect_value(t, session.pending_cmd_count, 1)
	testing.expect_value(t, session.pending_cmds[0].sequence, u32(6))
}

@(test)
test_server_command_queue_overflow_keeps_newest :: proc(t: ^testing.T) {
	session: ClientSession
	for sequence: u32 = 1; sequence <= SERVER_PENDING_USER_CMDS + 5; sequence += 1 {
		server_enqueue_user_cmd(&session, test_server_user_cmd(sequence))
	}

	testing.expect_value(t, session.pending_cmd_count, SERVER_PENDING_USER_CMDS)
	testing.expect_value(t, session.pending_cmds[0].sequence, u32(6))
	testing.expect_value(t, session.pending_cmds[SERVER_PENDING_USER_CMDS - 1].sequence, u32(SERVER_PENDING_USER_CMDS + 5))
}

@(test)
test_server_command_queue_batch_deduplicates :: proc(t: ^testing.T) {
	state: ServerState
	cmds := protocol.User_Cmds{count = 4}
	cmds.cmds[0] = test_server_user_cmd(2)
	cmds.cmds[1] = test_server_user_cmd(1)
	cmds.cmds[2] = test_server_user_cmd(2)
	cmds.cmds[3] = test_server_user_cmd(3)

	server_enqueue_user_cmds(&state, 0, cmds)
	session := &state.sessions[0]
	testing.expect_value(t, session.pending_cmd_count, 3)
	testing.expect_value(t, session.pending_cmds[0].sequence, u32(1))
	testing.expect_value(t, session.pending_cmds[1].sequence, u32(2))
	testing.expect_value(t, session.pending_cmds[2].sequence, u32(3))
}

@(test)
test_server_command_queue_empty_consume_uses_no_new_command :: proc(t: ^testing.T) {
	session := ClientSession{last_input_cmd = test_server_user_cmd(9), has_last_input_cmd = true}
	_, ok := server_consume_user_cmds(&session)
	testing.expect(t, !ok, "empty queue should not produce a new command")
	testing.expect_value(t, session.last_processed_cmd_sequence, u32(0))
	testing.expect(t, session.has_last_input_cmd, "fallback input should remain available")
}

test_server_user_cmd :: proc(sequence: u32) -> protocol.User_Cmd {
	return {
		sequence = sequence,
		client_tick = sequence,
		move_forward = f32(sequence),
		move_right = -f32(sequence),
		yaw = f32(sequence) * 0.25,
		pitch = -f32(sequence) * 0.125,
	}
}

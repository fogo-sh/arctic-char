package protocol

import "core:testing"

@(test)
test_client_hello_round_trips_map_name :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	packet, ok := write_client_hello(buffer[:], "test", 0x12345678)
	testing.expect(t, ok, "client hello should fit")

	header, header_err := parse_header(packet)
	testing.expect_value(t, header_err, Parse_Error.None)
	testing.expect_value(t, header.kind, Packet_Kind.Client_Hello)

	hello, payload_err := parse_hello_payload(packet, header)
	testing.expect_value(t, payload_err, Parse_Error.None)
	hello_map := hello.map_name
	testing.expect_value(t, map_name_string(&hello_map), "test")
	testing.expect_value(t, hello.content_id, u32(0x12345678))
}

@(test)
test_parse_packet_enforces_channel :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	packet, ok := write_client_hello(buffer[:], "test", 0)
	testing.expect(t, ok, "client hello should fit")

	_, err := parse_packet(packet, CHANNEL_USER_CMDS)
	testing.expect_value(t, err, Parse_Error.Bad_Channel)
}

@(test)
test_parse_packet_decodes_server_hello :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	packet, ok := write_server_hello(buffer[:], "test", 7)
	testing.expect(t, ok, "server hello should fit")

	parsed, err := parse_packet(packet, CHANNEL_CONTROL)
	testing.expect_value(t, err, Parse_Error.None)
	testing.expect_value(t, parsed.header.kind, Packet_Kind.Server_Hello)
	parsed_map := parsed.hello.map_name
	testing.expect_value(t, map_name_string(&parsed_map), "test")
	testing.expect_value(t, parsed.hello.content_id, u32(7))
}

@(test)
test_server_reject_round_trips_reason :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	packet, ok := write_server_reject(buffer[:], .Map_Mismatch)
	testing.expect(t, ok, "server reject should fit")

	header, header_err := parse_header(packet)
	testing.expect_value(t, header_err, Parse_Error.None)
	testing.expect_value(t, header.kind, Packet_Kind.Server_Reject)

	reason, payload_err := parse_reject_payload(packet, header)
	testing.expect_value(t, payload_err, Parse_Error.None)
	testing.expect_value(t, reason, Reject_Reason.Map_Mismatch)
}

@(test)
test_parse_packet_rejects_invalid_kind :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	packet, ok := write_client_hello(buffer[:], "test", 0)
	testing.expect(t, ok, "client hello should fit")
	packet[6] = 255

	_, err := parse_packet(packet, CHANNEL_CONTROL)
	testing.expect_value(t, err, Parse_Error.Bad_Kind)
}

@(test)
test_parse_packet_rejects_invalid_reject_reason :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	packet, ok := write_server_reject(buffer[:], .Map_Mismatch)
	testing.expect(t, ok, "server reject should fit")
	packet[HEADER_SIZE] = 255

	_, err := parse_packet(packet, CHANNEL_CONTROL)
	testing.expect_value(t, err, Parse_Error.Bad_Payload)
}

@(test)
test_header_rejects_bad_payload_length :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	packet, ok := write_client_hello(buffer[:], "test", 0)
	testing.expect(t, ok, "client hello should fit")
	packet = packet[:len(packet) - 1]

	_, err := parse_header(packet)
	testing.expect_value(t, err, Parse_Error.Bad_Length)
}

@(test)
test_user_cmd_round_trips :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	written := User_Cmd{
		sequence = 42,
		client_tick = 99,
		buttons = BUTTON_JUMP,
		move_forward = 1,
		move_right = -0.5,
		yaw = 1.25,
		pitch = -0.25,
	}
	packet, ok := write_user_cmd(buffer[:], written)
	testing.expect(t, ok, "user cmd should fit")

	header, header_err := parse_header(packet)
	testing.expect_value(t, header_err, Parse_Error.None)
	testing.expect_value(t, header.kind, Packet_Kind.User_Cmd)

	parsed, payload_err := parse_user_cmd_payload(packet, header)
	testing.expect_value(t, payload_err, Parse_Error.None)
	testing.expect_value(t, parsed.sequence, written.sequence)
	testing.expect_value(t, parsed.client_tick, written.client_tick)
	testing.expect_value(t, parsed.buttons, written.buttons)
	testing.expect_value(t, parsed.move_forward, written.move_forward)
	testing.expect_value(t, parsed.move_right, written.move_right)
	testing.expect_value(t, parsed.yaw, written.yaw)
	testing.expect_value(t, parsed.pitch, written.pitch)
}

@(test)
test_server_player_state_round_trips :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	written := Server_Player_State{player_id = 3, position = {1, 2, 3}, yaw = 1.5}
	packet, ok := write_server_player_state(buffer[:], written)
	testing.expect(t, ok, "server player state should fit")

	parsed, err := parse_packet(packet, CHANNEL_SNAPSHOTS)
	testing.expect_value(t, err, Parse_Error.None)
	testing.expect_value(t, parsed.header.kind, Packet_Kind.Server_Player_State)
	testing.expect_value(t, parsed.player_state.player_id, written.player_id)
	testing.expect_value(t, parsed.player_state.position, written.position)
	testing.expect_value(t, parsed.player_state.yaw, written.yaw)
}

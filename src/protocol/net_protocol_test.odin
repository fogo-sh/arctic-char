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
	packet, ok := write_server_hello(buffer[:], "test", 7, 3)
	testing.expect(t, ok, "server hello should fit")

	parsed, err := parse_packet(packet, CHANNEL_CONTROL)
	testing.expect_value(t, err, Parse_Error.None)
	testing.expect_value(t, parsed.header.kind, Packet_Kind.Server_Hello)
	parsed_map := parsed.hello.map_name
	testing.expect_value(t, map_name_string(&parsed_map), "test")
	testing.expect_value(t, parsed.hello.content_id, u32(7))
	testing.expect_value(t, parsed.hello.player_id, u32(3))
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
test_header_rejects_bad_magic_and_version :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	packet, ok := write_client_hello(buffer[:], "test", 0)
	testing.expect(t, ok, "client hello should fit")

	bad_magic := packet[:]
	bad_magic[0] = 0
	_, magic_err := parse_header(bad_magic)
	testing.expect_value(t, magic_err, Parse_Error.Bad_Magic)

	packet, ok = write_client_hello(buffer[:], "test", 0)
	testing.expect(t, ok, "client hello should fit")
	bad_version := packet[:]
	bad_version[4] = 0xff
	bad_version[5] = 0xff
	_, version_err := parse_header(bad_version)
	testing.expect_value(t, version_err, Parse_Error.Bad_Version)
}

@(test)
test_user_cmds_round_trips :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	written := User_Cmds{count = 2}
	written.cmds[0] = User_Cmd{
		sequence = 42,
		client_tick = 99,
		buttons = BUTTON_JUMP,
		move_forward = 1,
		move_right = -0.5,
		yaw = 1.25,
		pitch = -0.25,
	}
	written.cmds[1] = User_Cmd{
		sequence = 43,
		client_tick = 100,
		buttons = 0,
		move_forward = 0.25,
		move_right = 0.75,
		yaw = 1.5,
		pitch = 0.125,
	}
	packet, ok := write_user_cmds(buffer[:], written)
	testing.expect(t, ok, "user cmds should fit")

	header, header_err := parse_header(packet)
	testing.expect_value(t, header_err, Parse_Error.None)
	testing.expect_value(t, header.kind, Packet_Kind.User_Cmd)

	parsed, payload_err := parse_user_cmds_payload(packet, header)
	testing.expect_value(t, payload_err, Parse_Error.None)
	testing.expect_value(t, parsed.count, written.count)
	testing.expect_value(t, parsed.cmds[0].sequence, written.cmds[0].sequence)
	testing.expect_value(t, parsed.cmds[0].client_tick, written.cmds[0].client_tick)
	testing.expect_value(t, parsed.cmds[0].buttons, written.cmds[0].buttons)
	testing.expect_value(t, parsed.cmds[0].move_forward, written.cmds[0].move_forward)
	testing.expect_value(t, parsed.cmds[0].move_right, written.cmds[0].move_right)
	testing.expect_value(t, parsed.cmds[0].yaw, written.cmds[0].yaw)
	testing.expect_value(t, parsed.cmds[0].pitch, written.cmds[0].pitch)
	testing.expect_value(t, parsed.cmds[1].sequence, written.cmds[1].sequence)
	testing.expect_value(t, parsed.cmds[1].client_tick, written.cmds[1].client_tick)
	testing.expect_value(t, parsed.cmds[1].buttons, written.cmds[1].buttons)
	testing.expect_value(t, parsed.cmds[1].move_forward, written.cmds[1].move_forward)
	testing.expect_value(t, parsed.cmds[1].move_right, written.cmds[1].move_right)
	testing.expect_value(t, parsed.cmds[1].yaw, written.cmds[1].yaw)
	testing.expect_value(t, parsed.cmds[1].pitch, written.cmds[1].pitch)
}

@(test)
test_user_cmds_reject_invalid_count :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	cmds := User_Cmds{count = 1}
	cmds.cmds[0] = test_user_cmd(1)
	packet, ok := write_user_cmds(buffer[:], cmds)
	testing.expect(t, ok, "user cmds should fit")
	header, header_err := parse_header(packet)
	testing.expect_value(t, header_err, Parse_Error.None)

	packet[HEADER_SIZE] = 0
	_, zero_err := parse_user_cmds_payload(packet, header)
	testing.expect_value(t, zero_err, Parse_Error.Bad_Payload)

	packet[HEADER_SIZE] = byte(MAX_USER_CMDS_PER_PACKET + 1)
	_, too_many_err := parse_user_cmds_payload(packet, header)
	testing.expect_value(t, too_many_err, Parse_Error.Bad_Payload)

	too_many := User_Cmds{count = MAX_USER_CMDS_PER_PACKET + 1}
	_, write_ok := write_user_cmds(buffer[:], too_many)
	testing.expect(t, !write_ok, "too many user cmds should be rejected before writing")
}

@(test)
test_user_cmds_reject_payload_length_mismatch :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	w := packet_writer(buffer[:HEADER_SIZE + USER_CMDS_HEADER_PAYLOAD_SIZE + USER_CMD_PAYLOAD_SIZE - 1])
	testing.expect(t, write_header(&w, Header{magic = PROTOCOL_MAGIC, version = PROTOCOL_VERSION, kind = .User_Cmd, payload_length = USER_CMDS_HEADER_PAYLOAD_SIZE + USER_CMD_PAYLOAD_SIZE - 1}))
	testing.expect(t, write_u8(&w, 1))
	packet := buffer[:HEADER_SIZE + USER_CMDS_HEADER_PAYLOAD_SIZE + USER_CMD_PAYLOAD_SIZE - 1]

	parsed, err := parse_packet(packet, CHANNEL_USER_CMDS)
	_ = parsed
	testing.expect_value(t, err, Parse_Error.Bad_Payload)
}

@(test)
test_server_snapshot_round_trips :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	written := Server_Snapshot{server_tick = 120, sequence = 4, last_processed_user_cmd = 43, player_count = 2, prop_count = 1, removed_prop_count = 1}
	written.players[0] = Server_Player_State{
		player_id = 3,
		position = {1, 2, 3},
		velocity = {0.5, -1.0, 2.0},
		yaw = 1.5,
		pitch = -0.25,
		grounded = true,
		ground_normal = {0, 1, 0},
	}
	written.players[1] = Server_Player_State{
		player_id = 7,
		position = {4, 5, 6},
		velocity = {-3, 0, 9},
		yaw = 2.5,
		pitch = 0.75,
		grounded = false,
		ground_normal = {0.1, 0.8, -0.2},
	}
	written.props[0] = Server_Prop_State{
		object_id = 42,
		position = {8, 9, 10},
		rotation = {0.1, 0.2, 0.3, 0.9},
	}
	written.removed_prop_ids[0] = 99
	packet, ok := write_server_snapshot(buffer[:], written)
	testing.expect(t, ok, "server snapshot should fit")

	parsed, err := parse_packet(packet, CHANNEL_SNAPSHOTS)
	testing.expect_value(t, err, Parse_Error.None)
	testing.expect_value(t, parsed.header.kind, Packet_Kind.Server_Snapshot)
	testing.expect_value(t, parsed.snapshot.server_tick, written.server_tick)
	testing.expect_value(t, parsed.snapshot.sequence, written.sequence)
	testing.expect_value(t, parsed.snapshot.last_processed_user_cmd, written.last_processed_user_cmd)
	testing.expect_value(t, parsed.snapshot.player_count, written.player_count)
	testing.expect_value(t, parsed.snapshot.prop_count, written.prop_count)
	testing.expect_value(t, parsed.snapshot.removed_prop_count, written.removed_prop_count)
	testing.expect_value(t, parsed.snapshot.players[0].player_id, written.players[0].player_id)
	testing.expect_value(t, parsed.snapshot.players[0].position, written.players[0].position)
	testing.expect_value(t, parsed.snapshot.players[0].velocity, written.players[0].velocity)
	testing.expect_value(t, parsed.snapshot.players[0].yaw, written.players[0].yaw)
	testing.expect_value(t, parsed.snapshot.players[0].pitch, written.players[0].pitch)
	testing.expect_value(t, parsed.snapshot.players[0].grounded, written.players[0].grounded)
	testing.expect_value(t, parsed.snapshot.players[0].ground_normal, written.players[0].ground_normal)
	testing.expect_value(t, parsed.snapshot.players[1].player_id, written.players[1].player_id)
	testing.expect_value(t, parsed.snapshot.players[1].position, written.players[1].position)
	testing.expect_value(t, parsed.snapshot.players[1].velocity, written.players[1].velocity)
	testing.expect_value(t, parsed.snapshot.players[1].yaw, written.players[1].yaw)
	testing.expect_value(t, parsed.snapshot.players[1].pitch, written.players[1].pitch)
	testing.expect_value(t, parsed.snapshot.players[1].grounded, written.players[1].grounded)
	testing.expect_value(t, parsed.snapshot.players[1].ground_normal, written.players[1].ground_normal)
	testing.expect_value(t, parsed.snapshot.props[0].object_id, written.props[0].object_id)
	testing.expect_value(t, parsed.snapshot.props[0].position, written.props[0].position)
	testing.expect_value(t, parsed.snapshot.props[0].rotation, written.props[0].rotation)
	testing.expect_value(t, parsed.snapshot.removed_prop_ids[0], written.removed_prop_ids[0])
}

@(test)
test_server_snapshot_rejects_invalid_player_count :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	snapshot := Server_Snapshot{player_count = MAX_SNAPSHOT_PLAYERS + 1}
	_, write_ok := write_server_snapshot(buffer[:], snapshot)
	testing.expect(t, !write_ok, "too many players should be rejected before writing")

	w := packet_writer(buffer[:HEADER_SIZE + SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE])
	testing.expect(t, write_header(&w, Header{magic = PROTOCOL_MAGIC, version = PROTOCOL_VERSION, kind = .Server_Snapshot, payload_length = SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE}))
	testing.expect(t, write_u32(&w, 1))
	testing.expect(t, write_u32(&w, 2))
	testing.expect(t, write_u32(&w, 3))
	testing.expect(t, write_u16(&w, MAX_SNAPSHOT_PLAYERS + 1))
	testing.expect(t, write_u16(&w, 0))
	testing.expect(t, write_u16(&w, 0))
	packet := writer_bytes(&w)

	_, err := parse_packet(packet, CHANNEL_SNAPSHOTS)
	testing.expect_value(t, err, Parse_Error.Bad_Payload)
}

@(test)
test_server_snapshot_rejects_invalid_prop_count :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	snapshot := Server_Snapshot{prop_count = MAX_SNAPSHOT_PROPS + 1}
	_, write_ok := write_server_snapshot(buffer[:], snapshot)
	testing.expect(t, !write_ok, "too many props should be rejected before writing")

	w := packet_writer(buffer[:HEADER_SIZE + SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE])
	testing.expect(t, write_header(&w, Header{magic = PROTOCOL_MAGIC, version = PROTOCOL_VERSION, kind = .Server_Snapshot, payload_length = SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE}))
	testing.expect(t, write_u32(&w, 1))
	testing.expect(t, write_u32(&w, 2))
	testing.expect(t, write_u32(&w, 3))
	testing.expect(t, write_u16(&w, 0))
	testing.expect(t, write_u16(&w, MAX_SNAPSHOT_PROPS + 1))
	testing.expect(t, write_u16(&w, 0))
	packet := writer_bytes(&w)

	_, err := parse_packet(packet, CHANNEL_SNAPSHOTS)
	testing.expect_value(t, err, Parse_Error.Bad_Payload)
}

@(test)
test_server_snapshot_rejects_invalid_removed_prop_count :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	snapshot := Server_Snapshot{removed_prop_count = MAX_REMOVED_PROPS + 1}
	_, write_ok := write_server_snapshot(buffer[:], snapshot)
	testing.expect(t, !write_ok, "too many removed props should be rejected before writing")

	w := packet_writer(buffer[:HEADER_SIZE + SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE])
	testing.expect(t, write_header(&w, Header{magic = PROTOCOL_MAGIC, version = PROTOCOL_VERSION, kind = .Server_Snapshot, payload_length = SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE}))
	testing.expect(t, write_u32(&w, 1))
	testing.expect(t, write_u32(&w, 2))
	testing.expect(t, write_u32(&w, 3))
	testing.expect(t, write_u16(&w, 0))
	testing.expect(t, write_u16(&w, 0))
	testing.expect(t, write_u16(&w, MAX_REMOVED_PROPS + 1))
	packet := writer_bytes(&w)

	_, err := parse_packet(packet, CHANNEL_SNAPSHOTS)
	testing.expect_value(t, err, Parse_Error.Bad_Payload)
}

@(test)
test_map_name_boundaries :: proc(t: ^testing.T) {
	buffer: [MAX_PACKET_SIZE]byte
	empty_packet, empty_ok := write_client_hello(buffer[:], "", 0)
	testing.expect(t, empty_ok, "empty map names are currently accepted")
	empty, empty_err := parse_packet(empty_packet, CHANNEL_CONTROL)
	testing.expect_value(t, empty_err, Parse_Error.None)
	empty_map := empty.hello.map_name
	testing.expect_value(t, map_name_string(&empty_map), "")

	max_name_bytes: [MAX_MAP_NAME_BYTES]byte
	for &b in max_name_bytes do b = 'a'
	max_name := transmute(string)max_name_bytes[:]
	max_packet, max_ok := write_client_hello(buffer[:], max_name, 0)
	testing.expect(t, max_ok, "max length map name should fit")
	max_parsed, max_err := parse_packet(max_packet, CHANNEL_CONTROL)
	testing.expect_value(t, max_err, Parse_Error.None)
	parsed_max_map := max_parsed.hello.map_name
	testing.expect_value(t, map_name_string(&parsed_max_map), max_name)

	over_name_bytes: [MAX_MAP_NAME_BYTES + 1]byte
	for &b in over_name_bytes do b = 'b'
	over_name := transmute(string)over_name_bytes[:]
	_, over_ok := write_client_hello(buffer[:], over_name, 0)
	testing.expect(t, !over_ok, "over max map name should be rejected")
}

test_user_cmd :: proc(sequence: u32) -> User_Cmd {
	return {
		sequence = sequence,
		client_tick = sequence + 100,
		buttons = BUTTON_JUMP,
		move_forward = f32(sequence),
		move_right = -f32(sequence),
		yaw = f32(sequence) * 0.25,
		pitch = -f32(sequence) * 0.125,
	}
}

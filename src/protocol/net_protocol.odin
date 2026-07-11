package protocol

PROTOCOL_MAGIC   :: u32(0x52484341) // "ACHR" in little-endian packet order.
PROTOCOL_VERSION :: u16(1)

CHANNEL_CONTROL   :: u8(0)
CHANNEL_USER_CMDS :: u8(1)
CHANNEL_SNAPSHOTS :: u8(2)
CHANNEL_DEBUG     :: u8(3)

MAX_PACKET_SIZE :: 4096
HEADER_SIZE     :: 9
MAX_MAP_NAME_BYTES :: 64
USER_CMD_PAYLOAD_SIZE :: 26
USER_CMDS_HEADER_PAYLOAD_SIZE :: 1
MAX_USER_CMDS_PER_PACKET :: 8
SERVER_PLAYER_STATE_PAYLOAD_SIZE :: 20
SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE :: 14
MAX_SNAPSHOT_PLAYERS :: 32

Packet_Kind :: enum u8 {
	Invalid,
	Client_Hello,
	Server_Hello,
	Server_Reject,
	User_Cmd,
	Server_Snapshot,
}

Reject_Reason :: enum u8 {
	None,
	Map_Mismatch,
	Content_Mismatch,
	Malformed_Hello,
}

BUTTON_JUMP :: u16(1 << 0)

Header :: struct {
	magic:          u32,
	version:        u16,
	kind:           Packet_Kind,
	payload_length: u16,
}

Map_Name :: struct {
	bytes:  [MAX_MAP_NAME_BYTES]byte,
	length: u8,
}

Hello_Payload :: struct {
	map_name:   Map_Name,
	content_id: u32,
	player_id:  u32,
}

Parsed_Packet :: struct {
	header:        Header,
	hello:         Hello_Payload,
	reject_reason: Reject_Reason,
	user_cmds:     User_Cmds,
	snapshot:      Server_Snapshot,
}

User_Cmd :: struct {
	sequence:     u32,
	client_tick:  u32,
	buttons:      u16,
	move_forward: f32,
	move_right:   f32,
	yaw:          f32,
	pitch:        f32,
}

User_Cmds :: struct {
	count: u8,
	cmds:  [MAX_USER_CMDS_PER_PACKET]User_Cmd,
}

Server_Player_State :: struct {
	player_id: u32,
	position:  [3]f32,
	yaw:       f32,
}

Server_Snapshot :: struct {
	server_tick: u32,
	sequence:    u32,
	last_processed_user_cmd: u32,
	player_count: u16,
	players:     [MAX_SNAPSHOT_PLAYERS]Server_Player_State,
}

Parse_Error :: enum {
	None,
	Too_Small,
	Bad_Magic,
	Bad_Version,
	Bad_Length,
	Bad_Kind,
	Bad_Channel,
	Bad_Payload,
}

write_client_hello :: proc(buffer: []byte, requested_map: string, content_id: u32) -> (packet: []byte, ok: bool) {
	return write_hello_packet(buffer, .Client_Hello, requested_map, content_id)
}

write_server_hello :: proc(buffer: []byte, map_name: string, content_id: u32, player_id: u32) -> (packet: []byte, ok: bool) {
	return write_hello_packet(buffer, .Server_Hello, map_name, content_id, player_id)
}

write_server_reject :: proc(buffer: []byte, reason: Reject_Reason) -> (packet: []byte, ok: bool) {
	payload_length := 1
	total_length := HEADER_SIZE + payload_length
	if len(buffer) < total_length {
		return nil, false
	}

	w := packet_writer(buffer[:total_length])
	write_header(&w, Header{
		magic = PROTOCOL_MAGIC,
		version = PROTOCOL_VERSION,
		kind = .Server_Reject,
		payload_length = u16(payload_length),
	}) or_return
	write_u8(&w, u8(reason)) or_return
	return writer_bytes(&w), w.err == .None
}

write_hello_packet :: proc(buffer: []byte, kind: Packet_Kind, map_name: string, content_id: u32, player_id := u32(0)) -> (packet: []byte, ok: bool) {
	if kind != .Client_Hello && kind != .Server_Hello {
		return nil, false
	}
	if len(map_name) > MAX_MAP_NAME_BYTES {
		return nil, false
	}

	payload_length := 4 + 1 + len(map_name)
	if kind == .Server_Hello {
		payload_length += 4
	}
	total_length := HEADER_SIZE + payload_length
	if len(buffer) < total_length {
		return nil, false
	}

	w := packet_writer(buffer[:total_length])
	write_header(&w, Header{
		magic = PROTOCOL_MAGIC,
		version = PROTOCOL_VERSION,
		kind = kind,
		payload_length = u16(payload_length),
	}) or_return
	write_u32(&w, content_id) or_return
	if kind == .Server_Hello {
		write_u32(&w, player_id) or_return
	}
	write_u8(&w, u8(len(map_name))) or_return
	write_bytes(&w, transmute([]byte)map_name) or_return
	return writer_bytes(&w), w.err == .None
}

expected_channel :: proc(kind: Packet_Kind) -> (channel: u8, ok: bool) {
	#partial switch kind {
	case .Client_Hello, .Server_Hello, .Server_Reject:
		return CHANNEL_CONTROL, true
	case .User_Cmd:
		return CHANNEL_USER_CMDS, true
	case .Server_Snapshot:
		return CHANNEL_SNAPSHOTS, true
	case:
		return 0, false
	}
}

parse_packet :: proc(packet: []byte, channel: u8) -> (parsed: Parsed_Packet, err: Parse_Error) {
	header: Header
	header, err = parse_header(packet)
	if err != .None {
		return {}, err
	}
	parsed.header = header

	expected, ok := expected_channel(header.kind)
	if !ok {
		return parsed, .Bad_Kind
	}
	if channel != expected {
		return parsed, .Bad_Channel
	}

	#partial switch header.kind {
	case .Client_Hello, .Server_Hello:
		parsed.hello, err = parse_hello_payload(packet, header)
	case .Server_Reject:
		parsed.reject_reason, err = parse_reject_payload(packet, header)
	case .User_Cmd:
		parsed.user_cmds, err = parse_user_cmds_payload(packet, header)
	case .Server_Snapshot:
		parsed.snapshot, err = parse_server_snapshot_payload(packet, header)
	case:
		err = .Bad_Kind
	}
	return parsed, err
}

write_user_cmd :: proc(buffer: []byte, cmd: User_Cmd) -> (packet: []byte, ok: bool) {
	cmds := User_Cmds{count = 1}
	cmds.cmds[0] = cmd
	return write_user_cmds(buffer, cmds)
}

write_user_cmds :: proc(buffer: []byte, cmds: User_Cmds) -> (packet: []byte, ok: bool) {
	if cmds.count == 0 || cmds.count > MAX_USER_CMDS_PER_PACKET {
		return nil, false
	}
	payload_length := USER_CMDS_HEADER_PAYLOAD_SIZE + int(cmds.count) * USER_CMD_PAYLOAD_SIZE
	total_length := HEADER_SIZE + payload_length
	if len(buffer) < total_length {
		return nil, false
	}

	w := packet_writer(buffer[:total_length])
	write_header(&w, Header{
		magic = PROTOCOL_MAGIC,
		version = PROTOCOL_VERSION,
		kind = .User_Cmd,
		payload_length = u16(payload_length),
	}) or_return
	write_u8(&w, cmds.count) or_return
	for i in 0..<int(cmds.count) {
		write_user_cmd_payload(&w, cmds.cmds[i]) or_return
	}
	return writer_bytes(&w), w.err == .None
}

write_server_snapshot :: proc(buffer: []byte, snapshot: Server_Snapshot) -> (packet: []byte, ok: bool) {
	if snapshot.player_count > MAX_SNAPSHOT_PLAYERS {
		return nil, false
	}
	payload_length := SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE + int(snapshot.player_count) * SERVER_PLAYER_STATE_PAYLOAD_SIZE
	total_length := HEADER_SIZE + payload_length
	if len(buffer) < total_length {
		return nil, false
	}

	w := packet_writer(buffer[:total_length])
	write_header(&w, Header{
		magic = PROTOCOL_MAGIC,
		version = PROTOCOL_VERSION,
		kind = .Server_Snapshot,
		payload_length = u16(payload_length),
	}) or_return
	write_u32(&w, snapshot.server_tick) or_return
	write_u32(&w, snapshot.sequence) or_return
	write_u32(&w, snapshot.last_processed_user_cmd) or_return
	write_u16(&w, snapshot.player_count) or_return
	for i in 0..<int(snapshot.player_count) {
		write_server_player_state_payload(&w, snapshot.players[i]) or_return
	}
	return writer_bytes(&w), w.err == .None
}

parse_header :: proc(packet: []byte) -> (header: Header, err: Parse_Error) {
	if len(packet) < HEADER_SIZE {
		return {}, .Too_Small
	}

	r := packet_reader(packet[:HEADER_SIZE])
	magic, _ := read_u32(&r)
	version, _ := read_u16(&r)
	kind, _ := read_u8(&r)
	payload_length, _ := read_u16(&r)
	header = Header{
		magic = magic,
		version = version,
		kind = cast(Packet_Kind)kind,
		payload_length = payload_length,
	}

	if header.magic != PROTOCOL_MAGIC {
		return header, .Bad_Magic
	}
	if header.version != PROTOCOL_VERSION {
		return header, .Bad_Version
	}
	if int(header.payload_length) != len(packet) - HEADER_SIZE {
		return header, .Bad_Length
	}
	if header.kind == .Invalid || int(header.kind) > int(Packet_Kind.Server_Snapshot) {
		return header, .Bad_Kind
	}

	return header, .None
}

parse_hello_payload :: proc(packet: []byte, header: Header) -> (hello: Hello_Payload, err: Parse_Error) {
	if header.kind != .Client_Hello && header.kind != .Server_Hello {
		return {}, .Bad_Kind
	}
	if header.payload_length < 5 {
		return {}, .Bad_Payload
	}

	payload := packet[HEADER_SIZE:]
	r := packet_reader(payload)
	ok: bool
	content_id: u32
	content_id, ok = read_u32(&r)
	if !ok do return {}, .Bad_Payload
	player_id := u32(0)
	if header.kind == .Server_Hello {
		player_id, ok = read_u32(&r)
		if !ok do return {}, .Bad_Payload
	}
	map_length_u8: u8
	map_length_u8, ok = read_u8(&r)
	if !ok do return {}, .Bad_Payload
	map_length := int(map_length_u8)
	expected_remaining := map_length
	if map_length > MAX_MAP_NAME_BYTES || expected_remaining != reader_remaining(&r) {
		return {}, .Bad_Payload
	}
	map_bytes: []byte
	map_bytes, ok = read_bytes(&r, map_length)
	if !ok || reader_remaining(&r) != 0 do return {}, .Bad_Payload

	map_name, map_ok := map_name_from_bytes(map_bytes)
	if !map_ok do return {}, .Bad_Payload
	return Hello_Payload{map_name = map_name, content_id = content_id, player_id = player_id}, .None
}

parse_reject_payload :: proc(packet: []byte, header: Header) -> (reason: Reject_Reason, err: Parse_Error) {
	if header.kind != .Server_Reject {
		return .None, .Bad_Kind
	}
	if header.payload_length != 1 {
		return .None, .Bad_Payload
	}

	r := packet_reader(packet[HEADER_SIZE:])
	reason_u8, ok := read_u8(&r)
	if !ok || reader_remaining(&r) != 0 do return .None, .Bad_Payload
	reason = cast(Reject_Reason)reason_u8
	if reason == .None || int(reason) > int(Reject_Reason.Malformed_Hello) {
		return .None, .Bad_Payload
	}
	return reason, .None
}

parse_user_cmds_payload :: proc(packet: []byte, header: Header) -> (cmds: User_Cmds, err: Parse_Error) {
	if header.kind != .User_Cmd {
		return {}, .Bad_Kind
	}
	if header.payload_length < USER_CMDS_HEADER_PAYLOAD_SIZE {
		return {}, .Bad_Payload
	}

	r := packet_reader(packet[HEADER_SIZE:])
	ok: bool
	cmds.count, ok = read_u8(&r)
	if !ok do return {}, .Bad_Payload
	if cmds.count == 0 || cmds.count > MAX_USER_CMDS_PER_PACKET || reader_remaining(&r) != int(cmds.count) * USER_CMD_PAYLOAD_SIZE {
		return {}, .Bad_Payload
	}
	for i in 0..<int(cmds.count) {
		cmds.cmds[i], ok = read_user_cmd_payload(&r)
		if !ok do return {}, .Bad_Payload
	}
	if reader_remaining(&r) != 0 do return {}, .Bad_Payload
	return cmds, .None
}

parse_server_snapshot_payload :: proc(packet: []byte, header: Header) -> (snapshot: Server_Snapshot, err: Parse_Error) {
	if header.kind != .Server_Snapshot {
		return {}, .Bad_Kind
	}
	if header.payload_length < SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE {
		return {}, .Bad_Payload
	}

	r := packet_reader(packet[HEADER_SIZE:])
	ok: bool
	snapshot.server_tick, ok = read_u32(&r)
	if !ok do return {}, .Bad_Payload
	snapshot.sequence, ok = read_u32(&r)
	if !ok do return {}, .Bad_Payload
	snapshot.last_processed_user_cmd, ok = read_u32(&r)
	if !ok do return {}, .Bad_Payload
	snapshot.player_count, ok = read_u16(&r)
	if !ok do return {}, .Bad_Payload
	if snapshot.player_count > MAX_SNAPSHOT_PLAYERS || reader_remaining(&r) != int(snapshot.player_count) * SERVER_PLAYER_STATE_PAYLOAD_SIZE {
		return {}, .Bad_Payload
	}
	for i in 0..<int(snapshot.player_count) {
		snapshot.players[i], ok = read_server_player_state_payload(&r)
		if !ok do return {}, .Bad_Payload
	}
	if !ok || reader_remaining(&r) != 0 do return {}, .Bad_Payload
	return snapshot, .None
}

write_user_cmd_payload :: proc(w: ^Packet_Writer, cmd: User_Cmd) -> bool {
	write_u32(w, cmd.sequence) or_return
	write_u32(w, cmd.client_tick) or_return
	write_u16(w, cmd.buttons) or_return
	write_f32(w, cmd.move_forward) or_return
	write_f32(w, cmd.move_right) or_return
	write_f32(w, cmd.yaw) or_return
	write_f32(w, cmd.pitch) or_return
	return true
}

read_user_cmd_payload :: proc(r: ^Packet_Reader) -> (cmd: User_Cmd, ok: bool) {
	cmd.sequence, ok = read_u32(r)
	if !ok do return {}, false
	cmd.client_tick, ok = read_u32(r)
	if !ok do return {}, false
	cmd.buttons, ok = read_u16(r)
	if !ok do return {}, false
	cmd.move_forward, ok = read_f32(r)
	if !ok do return {}, false
	cmd.move_right, ok = read_f32(r)
	if !ok do return {}, false
	cmd.yaw, ok = read_f32(r)
	if !ok do return {}, false
	cmd.pitch, ok = read_f32(r)
	if !ok do return {}, false
	return cmd, true
}

write_server_player_state_payload :: proc(w: ^Packet_Writer, state: Server_Player_State) -> bool {
	write_u32(w, state.player_id) or_return
	write_f32(w, state.position.x) or_return
	write_f32(w, state.position.y) or_return
	write_f32(w, state.position.z) or_return
	write_f32(w, state.yaw) or_return
	return true
}

read_server_player_state_payload :: proc(r: ^Packet_Reader) -> (state: Server_Player_State, ok: bool) {
	state.player_id, ok = read_u32(r)
	if !ok do return {}, false
	state.position.x, ok = read_f32(r)
	if !ok do return {}, false
	state.position.y, ok = read_f32(r)
	if !ok do return {}, false
	state.position.z, ok = read_f32(r)
	if !ok do return {}, false
	state.yaw, ok = read_f32(r)
	if !ok do return {}, false
	return state, true
}

map_name_from_string :: proc(value: string) -> (name: Map_Name, ok: bool) {
	return map_name_from_bytes(transmute([]byte)value)
}

map_name_from_bytes :: proc(data: []byte) -> (name: Map_Name, ok: bool) {
	if len(data) > MAX_MAP_NAME_BYTES {
		return {}, false
	}
	name.length = u8(len(data))
	copy(name.bytes[:], data)
	return name, true
}

map_name_string :: proc(name: ^Map_Name) -> string {
	return transmute(string)name.bytes[:name.length]
}

map_name_equals_string :: proc(name: ^Map_Name, value: string) -> bool {
	if int(name.length) != len(value) {
		return false
	}
	for i in 0..<int(name.length) {
		if name.bytes[i] != value[i] do return false
	}
	return true
}

write_header :: proc(w: ^Packet_Writer, header: Header) -> bool {
	write_u32(w, header.magic) or_return
	write_u16(w, header.version) or_return
	write_u8(w, u8(header.kind)) or_return
	write_u16(w, header.payload_length) or_return
	return true
}

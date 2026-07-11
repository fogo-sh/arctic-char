package protocol

import "core:testing"

@(test)
test_packet_writer_reader_round_trip_scalars :: proc(t: ^testing.T) {
	buffer: [32]byte
	w := packet_writer(buffer[:])
	testing.expect(t, write_u8(&w, 7), "write u8")
	testing.expect(t, write_u16(&w, 0x1234), "write u16")
	testing.expect(t, write_u32(&w, 0x89abcdef), "write u32")
	testing.expect(t, write_f32(&w, -1.25), "write f32")
	testing.expect_value(t, w.err, Packet_IO_Error.None)

	r := packet_reader(writer_bytes(&w))
	ok: bool
	u8_value: u8
	u8_value, ok = read_u8(&r)
	testing.expect(t, ok, "read u8")
	testing.expect_value(t, u8_value, u8(7))
	u16_value: u16
	u16_value, ok = read_u16(&r)
	testing.expect(t, ok, "read u16")
	testing.expect_value(t, u16_value, u16(0x1234))
	u32_value: u32
	u32_value, ok = read_u32(&r)
	testing.expect(t, ok, "read u32")
	testing.expect_value(t, u32_value, u32(0x89abcdef))
	f32_value: f32
	f32_value, ok = read_f32(&r)
	testing.expect(t, ok, "read f32")
	testing.expect_value(t, f32_value, f32(-1.25))
	testing.expect_value(t, reader_remaining(&r), 0)
}

@(test)
test_packet_writer_reports_out_of_space_and_stops :: proc(t: ^testing.T) {
	buffer: [1]byte
	w := packet_writer(buffer[:])
	testing.expect(t, !write_u16(&w, 1), "u16 should not fit")
	testing.expect_value(t, w.err, Packet_IO_Error.Out_Of_Space)
	testing.expect(t, !write_u8(&w, 1), "writer should stay failed")
}

@(test)
test_packet_reader_reports_out_of_data_and_stops :: proc(t: ^testing.T) {
	r := packet_reader([]byte{1})
	_, ok := read_u16(&r)
	testing.expect(t, !ok, "u16 should not read from one byte")
	testing.expect_value(t, r.err, Packet_IO_Error.Out_Of_Data)
	_, ok = read_u8(&r)
	testing.expect(t, !ok, "reader should stay failed")
}

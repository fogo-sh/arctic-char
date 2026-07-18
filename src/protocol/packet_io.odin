package protocol

Packet_IO_Error :: enum {
	None,
	Out_Of_Space,
	Out_Of_Data,
	Invalid_Length,
}

Packet_Writer :: struct {
	data: []byte,
	off:  int,
	err:  Packet_IO_Error,
}

Packet_Reader :: struct {
	data: []byte,
	off:  int,
	err:  Packet_IO_Error,
}

packet_writer :: proc(buffer: []byte) -> Packet_Writer {
	return Packet_Writer{data = buffer}
}

packet_reader :: proc(data: []byte) -> Packet_Reader {
	return Packet_Reader{data = data}
}

writer_bytes :: proc(w: ^Packet_Writer) -> []byte {
	return w.data[:w.off]
}

reader_remaining :: proc(r: ^Packet_Reader) -> int {
	return len(r.data) - r.off
}

write_u8 :: proc(w: ^Packet_Writer, value: u8) -> bool {
	if !writer_require(w, 1) do return false
	w.data[w.off] = byte(value)
	w.off += 1
	return true
}

write_u16 :: proc(w: ^Packet_Writer, value: u16) -> bool {
	if !writer_require(w, 2) do return false
	w.data[w.off + 0] = byte(value & 0xff)
	w.data[w.off + 1] = byte((value >> 8) & 0xff)
	w.off += 2
	return true
}

write_i16 :: proc(w: ^Packet_Writer, value: i16) -> bool {
	return write_u16(w, transmute(u16)value)
}

write_u32 :: proc(w: ^Packet_Writer, value: u32) -> bool {
	if !writer_require(w, 4) do return false
	w.data[w.off + 0] = byte(value & 0xff)
	w.data[w.off + 1] = byte((value >> 8) & 0xff)
	w.data[w.off + 2] = byte((value >> 16) & 0xff)
	w.data[w.off + 3] = byte((value >> 24) & 0xff)
	w.off += 4
	return true
}

write_f32 :: proc(w: ^Packet_Writer, value: f32) -> bool {
	return write_u32(w, transmute(u32)value)
}

write_bytes :: proc(w: ^Packet_Writer, data: []byte) -> bool {
	if !writer_require(w, len(data)) do return false
	copy(w.data[w.off:], data)
	w.off += len(data)
	return true
}

read_u8 :: proc(r: ^Packet_Reader) -> (value: u8, ok: bool) {
	if !reader_require(r, 1) do return 0, false
	value = u8(r.data[r.off])
	r.off += 1
	return value, true
}

read_u16 :: proc(r: ^Packet_Reader) -> (value: u16, ok: bool) {
	if !reader_require(r, 2) do return 0, false
	value = u16(r.data[r.off + 0]) | (u16(r.data[r.off + 1]) << 8)
	r.off += 2
	return value, true
}

read_i16 :: proc(r: ^Packet_Reader) -> (value: i16, ok: bool) {
	bits: u16
	bits, ok = read_u16(r)
	if !ok do return 0, false
	return transmute(i16)bits, true
}

read_u32 :: proc(r: ^Packet_Reader) -> (value: u32, ok: bool) {
	if !reader_require(r, 4) do return 0, false
	value = u32(r.data[r.off + 0]) |
		(u32(r.data[r.off + 1]) << 8) |
		(u32(r.data[r.off + 2]) << 16) |
		(u32(r.data[r.off + 3]) << 24)
	r.off += 4
	return value, true
}

read_f32 :: proc(r: ^Packet_Reader) -> (value: f32, ok: bool) {
	bits: u32
	bits, ok = read_u32(r)
	if !ok do return 0, false
	return transmute(f32)bits, true
}

read_bytes :: proc(r: ^Packet_Reader, count: int) -> (data: []byte, ok: bool) {
	if count < 0 {
		r.err = .Invalid_Length
		return nil, false
	}
	if !reader_require(r, count) do return nil, false
	data = r.data[r.off:r.off + count]
	r.off += count
	return data, true
}

writer_require :: proc(w: ^Packet_Writer, count: int) -> bool {
	if w.err != .None do return false
	if count < 0 {
		w.err = .Invalid_Length
		return false
	}
	if w.off + count > len(w.data) {
		w.err = .Out_Of_Space
		return false
	}
	return true
}

reader_require :: proc(r: ^Packet_Reader, count: int) -> bool {
	if r.err != .None do return false
	if count < 0 {
		r.err = .Invalid_Length
		return false
	}
	if r.off + count > len(r.data) {
		r.err = .Out_Of_Data
		return false
	}
	return true
}

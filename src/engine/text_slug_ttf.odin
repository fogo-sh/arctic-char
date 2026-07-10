package engine

import "core:c"
import "core:math"
import "core:os"
import stbtt "vendor:stb/truetype"

// ===================================================
// TTF font loading via stb_truetype.
//
// Loads TTF files, extracts glyph outlines as quadratic Bezier curves,
// and provides kerning data. All coordinates are in em-space (normalized
// so the full ascent-to-descent range is ~1.0).
// ===================================================

// Load a TTF font file from disk.
font_load :: proc(path: string) -> (font: Font, ok: bool) {
	data, read_err := os.read_entire_file(path, context.allocator)
	if read_err != nil {
		return {}, false
	}
	font.font_data = data

	info := &font.info
	if stbtt.InitFont(info, raw_data(data), 0) == false {
		delete(data)
		return {}, false
	}

	ascent_raw, descent_raw, line_gap_raw: c.int
	stbtt.GetFontVMetrics(info, &ascent_raw, &descent_raw, &line_gap_raw)

	units_per_em := f32(ascent_raw - descent_raw)
	font.em_scale = 1.0 / units_per_em
	font.ascent = f32(ascent_raw) * font.em_scale
	font.descent = f32(descent_raw) * font.em_scale
	font.line_gap = f32(line_gap_raw) * font.em_scale

	// Read sCapHeight from the OS/2 table for pixel-grid alignment.
	font.cap_height = read_cap_height(data, font.em_scale)

	return font, true
}

// Load a single glyph's outline and metrics.
font_load_glyph :: proc(font: ^Font, codepoint: rune) -> bool {
	// Already loaded?
	if existing, ok := &font.glyphs[codepoint]; ok && existing.valid {
		return true
	}

	// Initialize map on first use
	if font.glyphs == nil {
		font.glyphs = make(map[rune]Glyph_Data, INITIAL_GLYPH_CAPACITY)
	}

	font.glyphs[codepoint] = {}
	g := &font.glyphs[codepoint]

	info := &font.info

	glyph_index := stbtt.FindGlyphIndex(info, codepoint)
	if glyph_index == 0 && codepoint != 0 {
		return false
	}

	g.codepoint = codepoint
	g.glyph_index = c.int(glyph_index)

	advance_raw, lsb_raw: c.int
	stbtt.GetGlyphHMetrics(info, c.int(glyph_index), &advance_raw, &lsb_raw)
	g.advance_width = f32(advance_raw) * font.em_scale
	g.left_bearing = f32(lsb_raw) * font.em_scale

	x0, y0, x1, y1: c.int
	if stbtt.GetGlyphBox(info, c.int(glyph_index), &x0, &y0, &x1, &y1) == 0 {
		g.bbox_min = {f32(0), f32(0)}
		g.bbox_max = {g.advance_width, font.ascent - font.descent}
		g.valid = true
		return true
	}

	g.bbox_min = {f32(x0) * font.em_scale, f32(y0) * font.em_scale}
	g.bbox_max = {f32(x1) * font.em_scale, f32(y1) * font.em_scale}

	vertices: [^]stbtt.vertex
	num_vertices := stbtt.GetGlyphShape(info, c.int(glyph_index), &vertices)
	if num_vertices <= 0 {
		g.valid = true
		return true
	}
	defer stbtt.FreeShape(info, vertices)

	// stb_truetype vertex type constants
	STBTT_VMOVE :: 1
	STBTT_VLINE :: 2
	STBTT_VCURVE :: 3
	STBTT_VCUBIC :: 4

	verts := vertices[:num_vertices]
	for i := 0; i < len(verts); i += 1 {
		v := verts[i]

		switch v.type {
		case STBTT_VLINE:
			if i == 0 do continue
			prev := verts[i - 1]
			p1 := [2]f32{f32(prev.x) * font.em_scale, f32(prev.y) * font.em_scale}
			p3 := [2]f32{f32(v.x) * font.em_scale, f32(v.y) * font.em_scale}
			p2 := [2]f32{(p1.x + p3.x) * 0.5, (p1.y + p3.y) * 0.5}
			append(&g.curves, Bezier_Curve{p1, p2, p3})

		case STBTT_VCURVE:
			if i == 0 do continue
			prev := verts[i - 1]
			p1 := [2]f32{f32(prev.x) * font.em_scale, f32(prev.y) * font.em_scale}
			p2 := [2]f32{f32(v.cx) * font.em_scale, f32(v.cy) * font.em_scale}
			p3 := [2]f32{f32(v.x) * font.em_scale, f32(v.y) * font.em_scale}
			append(&g.curves, Bezier_Curve{p1, p2, p3})

		case STBTT_VCUBIC:
			if i == 0 do continue
			prev := verts[i - 1]
			cp0 := [2]f32{f32(prev.x) * font.em_scale, f32(prev.y) * font.em_scale}
			cp1 := [2]f32{f32(v.cx) * font.em_scale, f32(v.cy) * font.em_scale}
			cp2 := [2]f32{f32(v.cx1) * font.em_scale, f32(v.cy1) * font.em_scale}
			cp3 := [2]f32{f32(v.x) * font.em_scale, f32(v.y) * font.em_scale}
			cubic_to_quadratics(cp0, cp1, cp2, cp3, &g.curves, CUBIC_TO_QUAD_TOLERANCE)

		case STBTT_VMOVE:
			continue
		}
	}

	g.valid = true
	return true
}

// Get kerning adjustment between two glyphs (in em-space units).
font_get_kerning :: proc(font: ^Font, left, right: rune) -> f32 {
	gl := get_glyph(font, left)
	gr := get_glyph(font, right)
	if gl == nil || gr == nil do return 0

	kern_raw := stbtt.GetGlyphKernAdvance(&font.info, c.int(gl.glyph_index), c.int(gr.glyph_index))
	return f32(kern_raw) * font.em_scale
}

// Load all ASCII printable glyphs (32-126).
font_load_ascii :: proc(font: ^Font) -> int {
	return font_load_range(font, 32, 126)
}

// Load all glyphs in a codepoint range (inclusive).
// Returns the number of glyphs successfully loaded.
// Common ranges:
//   Latin-1 Supplement: 160–255 (accented characters: é, ñ, ü, etc.)
//   Latin Extended-A:   256–383 (Ş, ž, Ő, etc.)
//   Greek:              880–1023
//   Cyrillic:           1024–1279
//   Box Drawing:        9472–9599
font_load_range :: proc(font: ^Font, first, last: rune) -> int {
	loaded := 0
	for cp := first; cp <= last; cp += 1 {
		if font_load_glyph(font, cp) {
			loaded += 1
		}
	}
	return loaded
}

// ===================================================
// Cubic-to-quadratic Bezier conversion
// ===================================================

cubic_to_quadratics :: proc(
	p0, p1, p2, p3: [2]f32,
	output: ^[dynamic]Bezier_Curve,
	tolerance: f32,
	depth: int = 0,
) {
	MAX_DEPTH :: 8

	mid01 := (p0 + p1) * 0.5
	mid12 := (p1 + p2) * 0.5
	mid23 := (p2 + p3) * 0.5
	mid012 := (mid01 + mid12) * 0.5
	mid123 := (mid12 + mid23) * 0.5
	cubic_mid := (mid012 + mid123) * 0.5

	q1 := (p1 * 3.0 - p0 + p2 * 3.0 - p3) * 0.25
	quad_mid := (p0 + q1 * 2.0 + p3) * 0.25

	err := cubic_mid - quad_mid
	error_sq := err.x * err.x + err.y * err.y

	if error_sq <= tolerance * tolerance || depth >= MAX_DEPTH {
		append(output, Bezier_Curve{p0, q1, p3})
		return
	}

	cubic_to_quadratics(p0, mid01, mid012, cubic_mid, output, tolerance, depth + 1)
	cubic_to_quadratics(cubic_mid, mid123, mid23, p3, output, tolerance, depth + 1)
}

// ===================================================
// sCapHeight reading and pixel-grid alignment
// ===================================================

// Snap a desired font size (in pixels) to the nearest value where
// cap-height lands exactly on a pixel boundary. This eliminates the
// soft gray edge at the top of capital letters (H, E, T, I, etc.).
//
// Returns the original size unchanged if the font has no cap_height data.
// For animated/zooming text, skip snapping — it would cause visible size jumps.
//
// dpi: scale factor for HiDPI displays (1.0 = standard, 2.0 = Retina).
// The snap is computed in device pixels, then divided back by dpi.
font_snap_size :: proc(font: ^Font, target_size: f32, dpi: f32 = 1.0) -> f32 {
	if font.cap_height <= 0 do return target_size
	effective_dpi := dpi if dpi > 0 else 1.0

	// Cap height in device pixels at the target size
	cap_pixels := target_size * effective_dpi * font.cap_height

	// Snap to nearest integer
	snapped_cap := math.round(cap_pixels)
	if snapped_cap < 1 do snapped_cap = 1

	return snapped_cap / (font.cap_height * effective_dpi)
}

// Read sCapHeight from the font's OS/2 table.
// Falls back to measuring the 'H' glyph bbox if OS/2 version < 2.
// Returns 0 if neither source is available.
@(private = "file")
read_cap_height :: proc(font_data: []u8, em_scale: f32) -> f32 {
	// The TTF file starts with an offset table:
	//   uint32 sfVersion
	//   uint16 numTables
	//   uint16 searchRange, entrySelector, rangeShift
	// Then numTables table directory entries of 16 bytes each:
	//   uint32 tag, uint32 checksum, uint32 offset, uint32 length

	if len(font_data) < 12 do return 0

	num_tables := read_u16_be(font_data, 4)

	// Scan table directory for 'OS/2' tag (0x4F532F32)
	for i in 0 ..< int(num_tables) {
		entry_offset := 12 + i * 16
		if entry_offset + 16 > len(font_data) do break

		tag := read_u32_be(font_data, entry_offset)
		if tag != 0x4F532F32 do continue // Not 'OS/2'

		table_offset := int(read_u32_be(font_data, entry_offset + 8))
		table_length := int(read_u32_be(font_data, entry_offset + 12))

		// sCapHeight is at byte offset 88, requires OS/2 version >= 2
		if table_length < 90 do return 0
		if table_offset + 90 > len(font_data) do return 0

		version := read_u16_be(font_data, table_offset)
		if version < 2 do return 0 // sCapHeight not present in v0/v1

		raw := read_i16_be(font_data, table_offset + 88)
		if raw <= 0 do return 0

		return f32(raw) * em_scale
	}

	// No OS/2 table — try measuring the 'H' glyph as fallback
	return cap_height_from_h_glyph(font_data, em_scale)
}

// Measure the 'H' glyph's top edge as a cap-height fallback.
@(private = "file")
cap_height_from_h_glyph :: proc(font_data: []u8, em_scale: f32) -> f32 {
	info: stbtt.fontinfo
	if !stbtt.InitFont(&info, raw_data(font_data), 0) do return 0

	glyph := stbtt.FindGlyphIndex(&info, 'H')
	if glyph == 0 do return 0

	x0, y0, x1, y1: c.int
	if stbtt.GetGlyphBox(&info, c.int(glyph), &x0, &y0, &x1, &y1) == 0 do return 0

	return f32(y1) * em_scale
}

// Big-endian byte readers for raw TTF data.
@(private = "file")
read_u16_be :: proc(data: []u8, offset: int) -> u16 {
	return u16(data[offset]) << 8 | u16(data[offset + 1])
}

@(private = "file")
read_u32_be :: proc(data: []u8, offset: int) -> u32 {
	return u32(data[offset]) << 24 | u32(data[offset + 1]) << 16 |
	       u32(data[offset + 2]) << 8 | u32(data[offset + 3])
}

@(private = "file")
read_i16_be :: proc(data: []u8, offset: int) -> i16 {
	return cast(i16)read_u16_be(data, offset)
}

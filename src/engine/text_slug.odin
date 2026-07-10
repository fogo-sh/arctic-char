package engine

// Minimal Slug-style CPU text data for the SDL GPU text renderer.
// This owns font glyph outlines, band acceleration data, packed atlas data,
// and one frame's glyph vertices. GPU upload/draw lives in renderer_text.odin.

import stbtt "vendor:stb/truetype"

TEXT_BAND_TEXTURE_WIDTH_LOG2 :: 12
TEXT_BAND_TEXTURE_WIDTH :: 1 << TEXT_BAND_TEXTURE_WIDTH_LOG2

TEXT_INITIAL_GLYPH_CAPACITY :: 256
TEXT_MAX_GLYPH_QUADS :: 4096
TEXT_VERTICES_PER_QUAD :: 4
TEXT_INDICES_PER_QUAD :: 6
TEXT_MAX_GLYPH_VERTICES :: TEXT_MAX_GLYPH_QUADS * TEXT_VERTICES_PER_QUAD
TEXT_MAX_GLYPH_INDICES :: TEXT_MAX_GLYPH_QUADS * TEXT_INDICES_PER_QUAD
TEXT_DILATION_SCALE :: f32(1.0)
TEXT_CUBIC_TO_QUAD_TOLERANCE :: f32(0.001)

Text_Vertex :: struct {
	pos: [4]f32,
	tex: [4]f32,
	jac: [4]f32,
	bnd: [4]f32,
	col: Color,
}

Text_Bezier_Curve :: struct {
	p1: [2]f32,
	p2: [2]f32,
	p3: [2]f32,
}

Text_Band :: struct {
	curve_count: u16,
	data_offset: u16,
}

Text_Glyph :: struct {
	bbox_min:      [2]f32,
	bbox_max:      [2]f32,
	advance_width: f32,
	left_bearing:  f32,

	curves:        [dynamic]Text_Bezier_Curve,
	h_bands:       [dynamic]Text_Band,
	v_bands:       [dynamic]Text_Band,
	h_curve_lists: [dynamic]u16,
	v_curve_lists: [dynamic]u16,

	curve_tex_x:   u16,
	curve_tex_y:   u16,
	band_tex_x:    u16,
	band_tex_y:    u16,
	band_max_x:    u16,
	band_max_y:    u16,
	band_scale:    [2]f32,
	band_offset:   [2]f32,

	codepoint:     rune,
	glyph_index:   i32,
	valid:         bool,
}

Text_Font :: struct {
	info:       stbtt.fontinfo,
	font_data:  []u8,
	ascent:     f32,
	descent:    f32,
	line_gap:   f32,
	em_scale:   f32,
	cap_height: f32,
	glyphs:     map[rune]Text_Glyph,
}

Text_Texture_Pack :: struct {
	curve_data:   [dynamic][4]u16,
	curve_width:  u32,
	curve_height: u32,
	band_data:    [dynamic][2]u16,
	band_width:   u32,
	band_height:  u32,
}

Text_Context :: struct {
	font:       Text_Font,
	font_loaded: bool,
	vertices:   [TEXT_MAX_GLYPH_VERTICES]Text_Vertex,
	quad_count: u32,
}

text_begin :: proc(ctx: ^Text_Context) {
	ctx.quad_count = 0
}

text_vertex_count :: proc(ctx: ^Text_Context) -> u32 {
	return ctx.quad_count * TEXT_VERTICES_PER_QUAD
}

text_register_font :: proc(ctx: ^Text_Context, font: Text_Font) {
	ctx.font = font
	ctx.font_loaded = true
}

text_glyph :: proc(font: ^Text_Font, ch: rune) -> ^Text_Glyph {
	g, ok := &font.glyphs[ch]
	if !ok || !g.valid do return nil
	return g
}

text_glyph_destroy :: proc(g: ^Text_Glyph) {
	delete(g.curves)
	delete(g.h_bands)
	delete(g.v_bands)
	delete(g.h_curve_lists)
	delete(g.v_curve_lists)
	g^ = {}
}

text_font_destroy :: proc(font: ^Text_Font) {
	for _, &g in font.glyphs {
		text_glyph_destroy(&g)
	}
	delete(font.glyphs)
	delete(font.font_data)
	font^ = {}
}

text_context_destroy :: proc(ctx: ^Text_Context) {
	if ctx.font_loaded {
		text_font_destroy(&ctx.font)
	}
	ctx^ = {}
}

text_pack_destroy :: proc(pack: ^Text_Texture_Pack) {
	delete(pack.curve_data)
	delete(pack.band_data)
	pack^ = {}
}

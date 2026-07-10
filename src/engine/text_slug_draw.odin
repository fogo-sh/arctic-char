package engine

// Text drawing and measurement: CPU-side vertex packing only.

text_measure :: proc(font: ^Text_Font, text: string, font_size: f32, use_kerning: bool = true) -> (width, height: f32) {
	pen_x: f32 = 0
	prev_rune: rune = 0

	for ch in text {
		g := text_glyph(font, ch)
		if g == nil {
			prev_rune = ch
			continue
		}
		if use_kerning && prev_rune != 0 {
			pen_x += text_font_get_kerning(font, prev_rune, ch) * font_size
		}
		pen_x += g.advance_width * font_size
		prev_rune = ch
	}

	return pen_x, (font.ascent - font.descent) * font_size
}

text_draw :: proc(ctx: ^Text_Context, text: string, x, y: f32, font_size: f32, color: Color, use_kerning: bool = true) {
	font := &ctx.font
	pen_x := x
	prev_rune: rune = 0

	for ch in text {
		g := text_glyph(font, ch)
		if g == nil {
			prev_rune = ch
			continue
		}

		if use_kerning && prev_rune != 0 {
			pen_x += text_font_get_kerning(font, prev_rune, ch) * font_size
		}

		glyph_x := pen_x + g.bbox_min.x * font_size
		glyph_y := y - g.bbox_max.y * font_size
		glyph_w := (g.bbox_max.x - g.bbox_min.x) * font_size
		glyph_h := (g.bbox_max.y - g.bbox_min.y) * font_size

		if len(g.curves) > 0 && ctx.quad_count < TEXT_MAX_GLYPH_QUADS {
			text_emit_glyph_quad(ctx, g, glyph_x, glyph_y, glyph_w, glyph_h, color)
		}

		pen_x += g.advance_width * font_size
		prev_rune = ch
	}
}

@(private = "package")
text_emit_glyph_quad :: proc(ctx: ^Text_Context, g: ^Text_Glyph, x, y, w, h: f32, color: Color) {
	base := ctx.quad_count * TEXT_VERTICES_PER_QUAD
	if base + TEXT_VERTICES_PER_QUAD > TEXT_MAX_GLYPH_VERTICES do return

	em_min := g.bbox_min
	em_max := g.bbox_max

	glyph_loc := transmute(f32)(u32(g.band_tex_x) | (u32(g.band_tex_y) << 16))
	band_max := transmute(f32)(u32(g.band_max_x) | (u32(g.band_max_y) << 16))

	em_w := em_max.x - em_min.x
	em_h := em_max.y - em_min.y
	jac_00 := em_w / w if w > 0 else 0
	jac_11 := -(em_h / h) if h > 0 else 0

	corners := [4][2]f32 {
		{x,     y},
		{x + w, y},
		{x + w, y + h},
		{x,     y + h},
	}

	normals := [4][2]f32 {
		{-TEXT_DILATION_SCALE, -TEXT_DILATION_SCALE},
		{ TEXT_DILATION_SCALE, -TEXT_DILATION_SCALE},
		{ TEXT_DILATION_SCALE,  TEXT_DILATION_SCALE},
		{-TEXT_DILATION_SCALE,  TEXT_DILATION_SCALE},
	}

	em_coords := [4][2]f32 {
		{em_min.x, em_max.y},
		{em_max.x, em_max.y},
		{em_max.x, em_min.y},
		{em_min.x, em_min.y},
	}

	for vi in 0..<4 {
		ctx.vertices[base + u32(vi)] = Text_Vertex {
			pos = {corners[vi].x, corners[vi].y, normals[vi].x, normals[vi].y},
			tex = {em_coords[vi].x, em_coords[vi].y, glyph_loc, band_max},
			jac = {jac_00, 0, 0, jac_11},
			bnd = {g.band_scale.x, g.band_scale.y, g.band_offset.x, g.band_offset.y},
			col = color,
		}
	}

	ctx.quad_count += 1
}

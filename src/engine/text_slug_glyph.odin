package engine

import "core:math"

@(private = "package")
text_glyph_process :: proc(g: ^Text_Glyph) {
	if len(g.curves) == 0 do return

	num_curves := len(g.curves)
	band_count := max(1, int(math.sqrt(f32(num_curves)) * 2.0))
	bbox_w := g.bbox_max.x - g.bbox_min.x
	bbox_h := g.bbox_max.y - g.bbox_min.y
	if bbox_w <= 0 || bbox_h <= 0 do return

	h_band_count := band_count
	v_band_count := band_count
	resize(&g.h_bands, h_band_count)
	resize(&g.v_bands, v_band_count)

	h_lists := make([][dynamic]u16, h_band_count)
	defer {
		for &list in h_lists do delete(list)
		delete(h_lists)
	}
	v_lists := make([][dynamic]u16, v_band_count)
	defer {
		for &list in v_lists do delete(list)
		delete(v_lists)
	}

	BAND_EPSILON :: f32(1.0 / 1024.0)
	for ci in 0..<num_curves {
		curve := &g.curves[ci]
		min_y := min(curve.p1.y, curve.p2.y, curve.p3.y) - BAND_EPSILON
		max_y := max(curve.p1.y, curve.p2.y, curve.p3.y) + BAND_EPSILON
		min_x := min(curve.p1.x, curve.p2.x, curve.p3.x) - BAND_EPSILON
		max_x := max(curve.p1.x, curve.p2.x, curve.p3.x) + BAND_EPSILON

		band_y_start := int(math.floor((min_y - g.bbox_min.y) / bbox_h * f32(h_band_count)))
		band_y_end := int(math.floor((max_y - g.bbox_min.y) / bbox_h * f32(h_band_count)))
		band_y_start = clamp(band_y_start, 0, h_band_count - 1)
		band_y_end = clamp(band_y_end, 0, h_band_count - 1)
		for bi in band_y_start..=band_y_end {
			append(&h_lists[bi], u16(ci))
		}

		band_x_start := int(math.floor((min_x - g.bbox_min.x) / bbox_w * f32(v_band_count)))
		band_x_end := int(math.floor((max_x - g.bbox_min.x) / bbox_w * f32(v_band_count)))
		band_x_start = clamp(band_x_start, 0, v_band_count - 1)
		band_x_end = clamp(band_x_end, 0, v_band_count - 1)
		for bi in band_x_start..=band_x_end {
			append(&v_lists[bi], u16(ci))
		}
	}

	for &list in h_lists do text_sort_curve_indices_by_max_x(list[:], g.curves[:])
	for &list in v_lists do text_sort_curve_indices_by_max_y(list[:], g.curves[:])

	clear(&g.h_curve_lists)
	for bi in 0..<h_band_count {
		g.h_bands[bi] = Text_Band{curve_count = u16(len(h_lists[bi])), data_offset = u16(len(g.h_curve_lists))}
		for ci in h_lists[bi] do append(&g.h_curve_lists, ci)
	}

	clear(&g.v_curve_lists)
	for bi in 0..<v_band_count {
		g.v_bands[bi] = Text_Band{curve_count = u16(len(v_lists[bi])), data_offset = u16(len(g.v_curve_lists))}
		for ci in v_lists[bi] do append(&g.v_curve_lists, ci)
	}

	g.band_max_x = u16(v_band_count - 1)
	g.band_max_y = u16(h_band_count - 1)
	g.band_scale = {f32(v_band_count) / bbox_w, f32(h_band_count) / bbox_h}
	g.band_offset = {
		-g.bbox_min.x * f32(v_band_count) / bbox_w,
		-g.bbox_min.y * f32(h_band_count) / bbox_h,
	}
}

@(private = "file")
text_sort_curve_indices_by_max_x :: proc(indices: []u16, curves: []Text_Bezier_Curve) {
	for i := 1; i < len(indices); i += 1 {
		key := indices[i]
		key_max_x := max(curves[key].p1.x, curves[key].p2.x, curves[key].p3.x)
		j := i - 1
		for j >= 0 {
			j_max_x := max(curves[indices[j]].p1.x, curves[indices[j]].p2.x, curves[indices[j]].p3.x)
			if j_max_x >= key_max_x do break
			indices[j + 1] = indices[j]
			j -= 1
		}
		indices[j + 1] = key
	}
}

@(private = "file")
text_sort_curve_indices_by_max_y :: proc(indices: []u16, curves: []Text_Bezier_Curve) {
	for i := 1; i < len(indices); i += 1 {
		key := indices[i]
		key_max_y := max(curves[key].p1.y, curves[key].p2.y, curves[key].p3.y)
		j := i - 1
		for j >= 0 {
			j_max_y := max(curves[indices[j]].p1.y, curves[indices[j]].p2.y, curves[indices[j]].p3.y)
			if j_max_y >= key_max_y do break
			indices[j + 1] = indices[j]
			j -= 1
		}
		indices[j + 1] = key
	}
}

text_font_process :: proc(font: ^Text_Font) -> Text_Texture_Pack {
	for _, &g in font.glyphs {
		if g.valid && len(g.curves) > 0 {
			text_glyph_process(&g)
		}
	}
	return text_pack_glyphs(font)
}

@(private = "file")
text_pack_glyphs :: proc(font: ^Text_Font) -> (result: Text_Texture_Pack) {
	curve_x: u32 = 0
	curve_y: u32 = 0
	band_x: u32 = 0
	band_y: u32 = 0

	for _, &g in font.glyphs {
		if !g.valid || len(g.curves) == 0 do continue

		num_curve_texels := u32(len(g.curves) * 2)
		if curve_x + num_curve_texels > TEXT_BAND_TEXTURE_WIDTH {
			for curve_x < TEXT_BAND_TEXTURE_WIDTH {
				append(&result.curve_data, [4]u16{0, 0, 0, 0})
				curve_x += 1
			}
			curve_x = 0
			curve_y += 1
		}

		g.curve_tex_x = u16(curve_x)
		g.curve_tex_y = u16(curve_y)
		for &curve in g.curves {
			append(&result.curve_data, [4]u16{text_f32_to_f16(curve.p1.x), text_f32_to_f16(curve.p1.y), text_f32_to_f16(curve.p2.x), text_f32_to_f16(curve.p2.y)})
			append(&result.curve_data, [4]u16{text_f32_to_f16(curve.p3.x), text_f32_to_f16(curve.p3.y), 0, 0})
			curve_x += 2
		}

		h_count := len(g.h_bands)
		v_count := len(g.v_bands)
		total_band_texels := u32(h_count + v_count + len(g.h_curve_lists) + len(g.v_curve_lists))
		if band_x + total_band_texels > TEXT_BAND_TEXTURE_WIDTH {
			for band_x < TEXT_BAND_TEXTURE_WIDTH {
				append(&result.band_data, [2]u16{0, 0})
				band_x += 1
			}
			band_x = 0
			band_y += 1
		}

		g.band_tex_x = u16(band_x)
		g.band_tex_y = u16(band_y)
		curve_list_base := u32(h_count + v_count)

		for bi in 0..<h_count {
			band := &g.h_bands[bi]
			append(&result.band_data, [2]u16{band.curve_count, u16(curve_list_base + u32(band.data_offset))})
		}
		for bi in 0..<v_count {
			band := &g.v_bands[bi]
			append(&result.band_data, [2]u16{band.curve_count, u16(curve_list_base + u32(len(g.h_curve_lists)) + u32(band.data_offset))})
		}
		for ci in g.h_curve_lists {
			append(&result.band_data, text_curve_texel_location(&g, ci))
		}
		for ci in g.v_curve_lists {
			append(&result.band_data, text_curve_texel_location(&g, ci))
		}

		band_x += total_band_texels
	}

	result.curve_width = TEXT_BAND_TEXTURE_WIDTH
	result.curve_height = max(curve_y + 1, 1)
	result.band_width = TEXT_BAND_TEXTURE_WIDTH
	result.band_height = max(band_y + 1, 1)
	resize(&result.curve_data, int(result.curve_width * result.curve_height))
	resize(&result.band_data, int(result.band_width * result.band_height))
	return result
}

@(private = "file")
text_curve_texel_location :: proc(g: ^Text_Glyph, curve_index: u16) -> [2]u16 {
	curve_texel_offset := u32(curve_index) * 2
	x := u32(g.curve_tex_x) + curve_texel_offset
	y := u32(g.curve_tex_y)
	y += x / TEXT_BAND_TEXTURE_WIDTH
	x = x % TEXT_BAND_TEXTURE_WIDTH
	return {u16(x), u16(y)}
}

@(private = "file")
text_f32_to_f16 :: proc(value: f32) -> u16 {
	bits := transmute(u32)value
	sign := (bits >> 16) & 0x8000
	exp := i32((bits >> 23) & 0xFF) - 127
	mant := bits & 0x007FFFFF

	if exp == 128 do return u16(sign | 0x7C00 | (mant != 0 ? 0x0200 : 0))
	if exp > 15 do return u16(sign | 0x7C00)
	if exp < -14 {
		if exp < -24 do return u16(sign)
		mant |= 0x00800000
		shift := u32(-exp - 14 + 13)
		return u16(sign | u32(mant >> shift))
	}
	return u16(sign | u32((exp + 15) << 10) | (mant >> 13))
}

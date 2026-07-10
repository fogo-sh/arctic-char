package engine

import "core:math"

// ===================================================
// Glyph processing: band generation and texture packing.
//
// Bands are the key spatial acceleration structure in Slug. Each glyph's
// bounding box is sliced into horizontal and vertical bands. Each band
// stores only the curve indices that overlap it. The fragment shader
// determines which band the current pixel falls in and only evaluates
// those curves, turning O(n) per-pixel into O(n/sqrt(n)) on average.
//
// After band generation, all glyph data is packed into two GPU textures:
//   - Curve texture (float16x4): control points, 2 texels per curve
//   - Band texture (uint16x2): band headers + curve index lists
// Both are 4096 texels wide; glyphs are packed left-to-right, wrapping rows.
// ===================================================

// Process a glyph: generate horizontal and vertical bands, sort curves.
@(private = "package")
glyph_process :: proc(g: ^Glyph_Data) {
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
		for &l in h_lists do delete(l)
		delete(h_lists)
	}
	v_lists := make([][dynamic]u16, v_band_count)
	defer {
		for &l in v_lists do delete(l)
		delete(v_lists)
	}

	BAND_EPSILON :: f32(1.0 / 1024.0) // Em-space overlap to prevent boundary misses

	for ci in 0 ..< num_curves {
		curve := &g.curves[ci]

		min_y := min(curve.p1.y, curve.p2.y, curve.p3.y) - BAND_EPSILON
		max_y := max(curve.p1.y, curve.p2.y, curve.p3.y) + BAND_EPSILON
		min_x := min(curve.p1.x, curve.p2.x, curve.p3.x) - BAND_EPSILON
		max_x := max(curve.p1.x, curve.p2.x, curve.p3.x) + BAND_EPSILON

		band_y_start := int(math.floor((min_y - g.bbox_min.y) / bbox_h * f32(h_band_count)))
		band_y_end := int(math.floor((max_y - g.bbox_min.y) / bbox_h * f32(h_band_count)))
		band_y_start = clamp(band_y_start, 0, h_band_count - 1)
		band_y_end = clamp(band_y_end, 0, h_band_count - 1)

		for bi in band_y_start ..= band_y_end {
			append(&h_lists[bi], u16(ci))
		}

		band_x_start := int(math.floor((min_x - g.bbox_min.x) / bbox_w * f32(v_band_count)))
		band_x_end := int(math.floor((max_x - g.bbox_min.x) / bbox_w * f32(v_band_count)))
		band_x_start = clamp(band_x_start, 0, v_band_count - 1)
		band_x_end = clamp(band_x_end, 0, v_band_count - 1)

		for bi in band_x_start ..= band_x_end {
			append(&v_lists[bi], u16(ci))
		}
	}

	for &list in h_lists {
		sort_curve_indices_by_max_x(list[:], g.curves[:])
	}
	for &list in v_lists {
		sort_curve_indices_by_max_y(list[:], g.curves[:])
	}

	clear(&g.h_curve_lists)
	for bi in 0 ..< h_band_count {
		g.h_bands[bi] = Band {
			curve_count = u16(len(h_lists[bi])),
			data_offset = u16(len(g.h_curve_lists)),
		}
		for ci in h_lists[bi] {
			append(&g.h_curve_lists, ci)
		}
	}

	clear(&g.v_curve_lists)
	for bi in 0 ..< v_band_count {
		g.v_bands[bi] = Band {
			curve_count = u16(len(v_lists[bi])),
			data_offset = u16(len(g.v_curve_lists)),
		}
		for ci in v_lists[bi] {
			append(&g.v_curve_lists, ci)
		}
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
sort_curve_indices_by_max_x :: proc(indices: []u16, curves: []Bezier_Curve) {
	for i := 1; i < len(indices); i += 1 {
		key := indices[i]
		key_max_x := max(curves[key].p1.x, curves[key].p2.x, curves[key].p3.x)
		j := i - 1
		for j >= 0 {
			j_max_x := max(
				curves[indices[j]].p1.x,
				curves[indices[j]].p2.x,
				curves[indices[j]].p3.x,
			)
			if j_max_x >= key_max_x do break
			indices[j + 1] = indices[j]
			j -= 1
		}
		indices[j + 1] = key
	}
}

@(private = "file")
sort_curve_indices_by_max_y :: proc(indices: []u16, curves: []Bezier_Curve) {
	for i := 1; i < len(indices); i += 1 {
		key := indices[i]
		key_max_y := max(curves[key].p1.y, curves[key].p2.y, curves[key].p3.y)
		j := i - 1
		for j >= 0 {
			j_max_y := max(
				curves[indices[j]].p1.y,
				curves[indices[j]].p2.y,
				curves[indices[j]].p3.y,
			)
			if j_max_y >= key_max_y do break
			indices[j + 1] = indices[j]
			j -= 1
		}
		indices[j + 1] = key
	}
}

// ===================================================
// Texture packing
// ===================================================

Texture_Pack_Result :: struct {
	curve_data:   [dynamic][4]u16,
	curve_width:  u32,
	curve_height: u32,
	band_data:    [dynamic][2]u16,
	band_width:   u32,
	band_height:  u32,
}

pack_result_destroy :: proc(r: ^Texture_Pack_Result) {
	delete(r.curve_data)
	delete(r.band_data)
}

// Pack all glyphs in a font into GPU texture data.
// Returns raw texture data that backends upload to the GPU.
@(private = "file")
pack_glyph_textures :: proc(font: ^Font) -> (result: Texture_Pack_Result) {
	curve_x: u32 = 0
	curve_y: u32 = 0
	band_x: u32 = 0
	band_y: u32 = 0

	for _, &g in font.glyphs {
		if !g.valid || len(g.curves) == 0 do continue

		num_curve_texels := u32(len(g.curves) * 2)

		if curve_x + num_curve_texels > BAND_TEXTURE_WIDTH {
			for curve_x < BAND_TEXTURE_WIDTH {
				append(&result.curve_data, [4]u16{0, 0, 0, 0})
				curve_x += 1
			}
			curve_x = 0
			curve_y += 1
		}

		g.curve_tex_x = u16(curve_x)
		g.curve_tex_y = u16(curve_y)

		for &curve in g.curves {
			append(
				&result.curve_data,
				[4]u16 {
					f32_to_f16(curve.p1.x),
					f32_to_f16(curve.p1.y),
					f32_to_f16(curve.p2.x),
					f32_to_f16(curve.p2.y),
				},
			)
			append(
				&result.curve_data,
				[4]u16{f32_to_f16(curve.p3.x), f32_to_f16(curve.p3.y), 0, 0},
			)
			curve_x += 2
		}

		h_count := len(g.h_bands)
		v_count := len(g.v_bands)
		total_band_texels := u32(h_count + v_count + len(g.h_curve_lists) + len(g.v_curve_lists))

		if band_x + total_band_texels > BAND_TEXTURE_WIDTH {
			for band_x < BAND_TEXTURE_WIDTH {
				append(&result.band_data, [2]u16{0, 0})
				band_x += 1
			}
			band_x = 0
			band_y += 1
		}

		g.band_tex_x = u16(band_x)
		g.band_tex_y = u16(band_y)

		curve_list_base := u32(h_count + v_count)

		for bi in 0 ..< h_count {
			band := &g.h_bands[bi]
			append(
				&result.band_data,
				[2]u16{band.curve_count, u16(curve_list_base + u32(band.data_offset))},
			)
		}

		for bi in 0 ..< v_count {
			band := &g.v_bands[bi]
			append(
				&result.band_data,
				[2]u16 {
					band.curve_count,
					u16(curve_list_base + u32(len(g.h_curve_lists)) + u32(band.data_offset)),
				},
			)
		}

		for ci in g.h_curve_lists {
			curve_texel_offset := u32(ci) * 2
			ctx_x := u32(g.curve_tex_x) + curve_texel_offset
			ctx_y := u32(g.curve_tex_y)
			ctx_y += ctx_x / BAND_TEXTURE_WIDTH
			ctx_x = ctx_x % BAND_TEXTURE_WIDTH
			append(&result.band_data, [2]u16{u16(ctx_x), u16(ctx_y)})
		}

		for ci in g.v_curve_lists {
			curve_texel_offset := u32(ci) * 2
			ctx_x := u32(g.curve_tex_x) + curve_texel_offset
			ctx_y := u32(g.curve_tex_y)
			ctx_y += ctx_x / BAND_TEXTURE_WIDTH
			ctx_x = ctx_x % BAND_TEXTURE_WIDTH
			append(&result.band_data, [2]u16{u16(ctx_x), u16(ctx_y)})
		}

		band_x += total_band_texels
	}

	result.curve_width = BAND_TEXTURE_WIDTH
	result.curve_height = max(curve_y + 1, 1)
	result.band_width = BAND_TEXTURE_WIDTH
	result.band_height = max(band_y + 1, 1)

	target_curve_texels := int(result.curve_width * result.curve_height)
	if len(result.curve_data) < target_curve_texels {
		resize(&result.curve_data, target_curve_texels)
	}

	target_band_texels := int(result.band_width * result.band_height)
	if len(result.band_data) < target_band_texels {
		resize(&result.band_data, target_band_texels)
	}

	return result
}

// ===================================================
// Float16 conversion (f32 -> f16 stored as u16)
// ===================================================

@(private = "file")
f32_to_f16 :: proc(value: f32) -> u16 {
	bits := transmute(u32)value

	sign := (bits >> 16) & 0x8000
	exp := i32((bits >> 23) & 0xFF) - 127
	mant := bits & 0x007FFFFF

	if exp == 128 {
		return u16(sign | 0x7C00 | (mant != 0 ? 0x0200 : 0))
	}

	if exp > 15 {
		return u16(sign | 0x7C00)
	}

	if exp < -14 {
		if exp < -24 do return u16(sign)
		mant |= 0x00800000
		shift := u32(-exp - 14 + 13)
		return u16(sign | u32(mant >> shift))
	}

	return u16(sign | u32((exp + 15) << 10) | (mant >> 13))
}

// Process all valid glyphs in a font and pack textures.
// Convenience proc that calls glyph_process on each glyph, then packs.
// For multi-font setups, prefer fonts_process_shared() for better performance.
font_process :: proc(font: ^Font) -> Texture_Pack_Result {
	for _, &g in font.glyphs {
		if g.valid && len(g.curves) > 0 {
			glyph_process(&g)
		}
	}
	return pack_glyph_textures(font)
}

// Process all registered fonts and pack their glyphs into a single shared
// texture atlas. Returns one Texture_Pack_Result covering all fonts.
// Sets ctx.shared_atlas = true, enabling free font interleaving and
// single-draw-call rendering in backends.
//
// Call this AFTER loading all fonts and their glyphs (font_load_ascii, etc.).
// The returned pack replaces any per-font packs — use upload_shared_textures
// on the backend instead of per-font upload_font_textures.
fonts_process_shared :: proc(ctx: ^Context) -> Texture_Pack_Result {
	// Process glyphs for all registered fonts
	for fi in 0 ..< MAX_FONT_SLOTS {
		if !ctx.font_loaded[fi] do continue
		font := &ctx.fonts[fi]
		for _, &g in font.glyphs {
			if g.valid && len(g.curves) > 0 {
				glyph_process(&g)
			}
		}
	}

	ctx.shared_atlas = true
	return pack_glyph_textures_shared(ctx.fonts[:], ctx.font_loaded[:])
}

// Pack glyphs from multiple fonts into a single pair of GPU textures.
// Each glyph's curve_tex_x/y and band_tex_x/y are set to point into
// the shared atlas, so vertex data works unchanged.
@(private = "file")
pack_glyph_textures_shared :: proc(
	fonts: []Font,
	font_loaded: []bool,
) -> (result: Texture_Pack_Result) {
	curve_x: u32 = 0
	curve_y: u32 = 0
	band_x: u32 = 0
	band_y: u32 = 0

	for fi in 0 ..< len(fonts) {
		if !font_loaded[fi] do continue
		font := &fonts[fi]

		for _, &g in font.glyphs {
			if !g.valid || len(g.curves) == 0 do continue

			num_curve_texels := u32(len(g.curves) * 2)

			if curve_x + num_curve_texels > BAND_TEXTURE_WIDTH {
				for curve_x < BAND_TEXTURE_WIDTH {
					append(&result.curve_data, [4]u16{0, 0, 0, 0})
					curve_x += 1
				}
				curve_x = 0
				curve_y += 1
			}

			g.curve_tex_x = u16(curve_x)
			g.curve_tex_y = u16(curve_y)

			for &curve in g.curves {
				append(
					&result.curve_data,
					[4]u16 {
						f32_to_f16(curve.p1.x),
						f32_to_f16(curve.p1.y),
						f32_to_f16(curve.p2.x),
						f32_to_f16(curve.p2.y),
					},
				)
				append(
					&result.curve_data,
					[4]u16{f32_to_f16(curve.p3.x), f32_to_f16(curve.p3.y), 0, 0},
				)
				curve_x += 2
			}

			h_count := len(g.h_bands)
			v_count := len(g.v_bands)
			total_band_texels := u32(h_count + v_count + len(g.h_curve_lists) + len(g.v_curve_lists))

			if band_x + total_band_texels > BAND_TEXTURE_WIDTH {
				for band_x < BAND_TEXTURE_WIDTH {
					append(&result.band_data, [2]u16{0, 0})
					band_x += 1
				}
				band_x = 0
				band_y += 1
			}

			g.band_tex_x = u16(band_x)
			g.band_tex_y = u16(band_y)

			curve_list_base := u32(h_count + v_count)

			for bi in 0 ..< h_count {
				band := &g.h_bands[bi]
				append(
					&result.band_data,
					[2]u16{band.curve_count, u16(curve_list_base + u32(band.data_offset))},
				)
			}

			for bi in 0 ..< v_count {
				band := &g.v_bands[bi]
				append(
					&result.band_data,
					[2]u16 {
						band.curve_count,
						u16(curve_list_base + u32(len(g.h_curve_lists)) + u32(band.data_offset)),
					},
				)
			}

			for ci in g.h_curve_lists {
				curve_texel_offset := u32(ci) * 2
				ctx_x := u32(g.curve_tex_x) + curve_texel_offset
				ctx_y := u32(g.curve_tex_y)
				ctx_y += ctx_x / BAND_TEXTURE_WIDTH
				ctx_x = ctx_x % BAND_TEXTURE_WIDTH
				append(&result.band_data, [2]u16{u16(ctx_x), u16(ctx_y)})
			}

			for ci in g.v_curve_lists {
				curve_texel_offset := u32(ci) * 2
				ctx_x := u32(g.curve_tex_x) + curve_texel_offset
				ctx_y := u32(g.curve_tex_y)
				ctx_y += ctx_x / BAND_TEXTURE_WIDTH
				ctx_x = ctx_x % BAND_TEXTURE_WIDTH
				append(&result.band_data, [2]u16{u16(ctx_x), u16(ctx_y)})
			}

			band_x += total_band_texels
		}
	}

	result.curve_width = BAND_TEXTURE_WIDTH
	result.curve_height = max(curve_y + 1, 1)
	result.band_width = BAND_TEXTURE_WIDTH
	result.band_height = max(band_y + 1, 1)

	target_curve_texels := int(result.curve_width * result.curve_height)
	if len(result.curve_data) < target_curve_texels {
		resize(&result.curve_data, target_curve_texels)
	}

	target_band_texels := int(result.band_width * result.band_height)
	if len(result.band_data) < target_band_texels {
		resize(&result.band_data, target_band_texels)
	}

	return result
}

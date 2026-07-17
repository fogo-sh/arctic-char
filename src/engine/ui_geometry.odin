package engine

import "core:math"

UI_GEOMETRY_CORNER_SEGMENTS :: 8

UiGeometryDrawKind :: enum {
	Geometry,
	ScissorStart,
	ScissorEnd,
}

UiGeometryDraw :: struct {
	kind:   UiGeometryDrawKind,
	start:  int,
	count:  int,
	bounds: [4]f32,
}

ui_command_scaled :: proc(command: UiCommand, scale: [2]f32) -> UiCommand {
	result := command
	result.bounds = ui_bounds_scaled(command.bounds, scale)
	corner_scale := min(scale.x, scale.y)
	result.corner_radii = command.corner_radii * corner_scale
	result.border_widths = {
		command.border_widths.x * scale.x,
		command.border_widths.y * scale.x,
		command.border_widths.z * scale.y,
		command.border_widths.w * scale.y,
	}
	result.font_size = command.font_size * corner_scale
	return result
}

ui_bounds_scaled :: proc(bounds: [4]f32, scale: [2]f32) -> [4]f32 {
	return {bounds.x * scale.x, bounds.y * scale.y, bounds.z * scale.x, bounds.w * scale.y}
}

ui_geometry_append_command :: proc(vertices: ^[dynamic]VertexData, draws: ^[dynamic]UiGeometryDraw, command: UiCommand) {
	start := len(vertices^)
	#partial switch command.kind {
	case .Rectangle:
		ui_geometry_append_rounded_rect(vertices, command.bounds, command.corner_radii, command.color)
	case .Border:
		ui_geometry_append_rounded_border(vertices, command.bounds, command.corner_radii, command.border_widths, command.color)
	case:
	}
	count := len(vertices^) - start
	if count > 0 {
		append(draws, UiGeometryDraw{kind = .Geometry, start = start, count = count})
	}
}

ui_geometry_append_rounded_rect :: proc(vertices: ^[dynamic]VertexData, bounds, radii, color: [4]f32) {
	if bounds.z <= 0 || bounds.w <= 0 do return
	points := ui_geometry_rounded_rect_points(bounds, ui_geometry_clamp_radii(radii, bounds))
	center := [2]f32{bounds.x + bounds.z * 0.5, bounds.y + bounds.w * 0.5}
	for i in 0..<len(points) {
		ui_geometry_append_triangle(vertices, center, points[i], points[(i + 1) % len(points)], color)
	}
}

ui_geometry_append_rounded_border :: proc(vertices: ^[dynamic]VertexData, bounds, radii, border, color: [4]f32) {
	if bounds.z <= 0 || bounds.w <= 0 do return
	if border.x <= 0 && border.y <= 0 && border.z <= 0 && border.w <= 0 do return
	outer_radii := ui_geometry_clamp_radii(radii, bounds)
	outer := ui_geometry_rounded_rect_points(bounds, outer_radii)
	inner_bounds := [4]f32{
		bounds.x + border.x,
		bounds.y + border.z,
		max(bounds.z - border.x - border.y, 0),
		max(bounds.w - border.z - border.w, 0),
	}
	if inner_bounds.z <= 0 || inner_bounds.w <= 0 {
		ui_geometry_append_rounded_rect(vertices, bounds, radii, color)
		return
	}
	inner_radii := ui_geometry_clamp_radii({
		max(outer_radii.x - min(border.x, border.z), 0),
		max(outer_radii.y - min(border.y, border.z), 0),
		max(outer_radii.z - min(border.x, border.w), 0),
		max(outer_radii.w - min(border.y, border.w), 0),
	}, inner_bounds)
	inner := ui_geometry_rounded_rect_points(inner_bounds, inner_radii)
	count := min(len(outer), len(inner))
	for i in 0..<count {
		next := (i + 1) % count
		ui_geometry_append_triangle(vertices, outer[i], outer[next], inner[next], color)
		ui_geometry_append_triangle(vertices, inner[next], inner[i], outer[i], color)
	}
}

ui_geometry_rounded_rect_points :: proc(bounds, radii: [4]f32) -> [4 * (UI_GEOMETRY_CORNER_SEGMENTS + 1)][2]f32 {
	points: [4 * (UI_GEOMETRY_CORNER_SEGMENTS + 1)][2]f32
	index := 0
	ui_geometry_append_corner_points(&points, &index, {bounds.x + radii.x, bounds.y + radii.x}, radii.x, math.PI, math.PI * 1.5)
	ui_geometry_append_corner_points(&points, &index, {bounds.x + bounds.z - radii.y, bounds.y + radii.y}, radii.y, math.PI * 1.5, math.PI * 2.0)
	ui_geometry_append_corner_points(&points, &index, {bounds.x + bounds.z - radii.w, bounds.y + bounds.w - radii.w}, radii.w, 0, math.PI * 0.5)
	ui_geometry_append_corner_points(&points, &index, {bounds.x + radii.z, bounds.y + bounds.w - radii.z}, radii.z, math.PI * 0.5, math.PI)
	return points
}

ui_geometry_append_corner_points :: proc(points: ^[4 * (UI_GEOMETRY_CORNER_SEGMENTS + 1)][2]f32, index: ^int, center: [2]f32, radius, angle0, angle1: f32) {
	for segment in 0..=UI_GEOMETRY_CORNER_SEGMENTS {
		t := f32(segment) / f32(UI_GEOMETRY_CORNER_SEGMENTS)
		angle := angle0 + (angle1 - angle0) * t
		points[index^] = {center.x + math.cos(angle) * radius, center.y + math.sin(angle) * radius}
		index^ += 1
	}
}

ui_geometry_append_triangle :: proc(vertices: ^[dynamic]VertexData, a, b, c: [2]f32, color: [4]f32) {
	if len(vertices^) + 3 > cap(vertices^) do return
	append(vertices,
		VertexData{pos = {a.x, a.y, 0}, color = color},
		VertexData{pos = {b.x, b.y, 0}, color = color},
		VertexData{pos = {c.x, c.y, 0}, color = color},
	)
}

ui_geometry_clamp_radii :: proc(radii, bounds: [4]f32) -> [4]f32 {
	limit := min(bounds.z, bounds.w) * 0.5
	return {min(radii.x, limit), min(radii.y, limit), min(radii.z, limit), min(radii.w, limit)}
}

package main

import "core:log"
import "core:math"
import "core:math/linalg"

MAP_PLANE_EPSILON :: f32(0.001)
MAP_VERTEX_EPSILON :: f32(0.01)
MAP_BASE_WINDING_MIN_SIZE :: f32(64.0)
MAP_MAX_WINDING_POINTS :: 64

MapWinding :: struct {
	points: [MAP_MAX_WINDING_POINTS]Vec3,
	count:  int,
}

create_map_mesh :: proc(qmap: ^QuakeMap, allocator := context.allocator) -> CpuMesh {
	vertices := make([dynamic]VertexData, 0, 8192, allocator)
	indices := make([dynamic]u32, 0, 24576, allocator)
	base_winding_size := map_mesh_base_winding_size(qmap)
	skipped_faces := 0
	emitted_faces := 0
	world_brushes := 0
	world_faces := 0

	for &entity in qmap.entities {
		classname, ok := map_entity_property(&entity, "classname")
		if !ok || classname != "worldspawn" {
			continue
		}

		for &brush in entity.brushes {
			world_brushes += 1
			world_faces += len(brush.faces)
			brush_emitted, brush_skipped := map_mesh_append_brush(&vertices, &indices, &brush, base_winding_size)
			emitted_faces += brush_emitted
			skipped_faces += brush_skipped
		}
	}

	map_mesh_assert_uploadable(vertices[:], indices[:])
	log.debugf(
		"Built map mesh: brushes=%d faces=%d emitted_faces=%d skipped_faces=%d vertices=%d indices=%d base_winding=%.2f",
		world_brushes,
		world_faces,
		emitted_faces,
		skipped_faces,
		len(vertices),
		len(indices),
		base_winding_size,
	)
	return CpuMesh{vertices = vertices[:], indices = indices[:], allocator = allocator}
}

map_mesh_base_winding_size :: proc(qmap: ^QuakeMap) -> f32 {
	mins := Vec3{math.F32_MAX, math.F32_MAX, math.F32_MAX}
	maxs := Vec3{-math.F32_MAX, -math.F32_MAX, -math.F32_MAX}
	point_count := 0

	for &entity in qmap.entities {
		for &brush in entity.brushes {
			for &face in brush.faces {
				for point in face.points {
					mins.x = min(mins.x, point.x)
					mins.y = min(mins.y, point.y)
					mins.z = min(mins.z, point.z)
					maxs.x = max(maxs.x, point.x)
					maxs.y = max(maxs.y, point.y)
					maxs.z = max(maxs.z, point.z)
					point_count += 1
				}
			}
		}
	}

	if point_count == 0 {
		return MAP_BASE_WINDING_MIN_SIZE
	}
	return max(linalg.length(maxs - mins) * 2, MAP_BASE_WINDING_MIN_SIZE)
}

map_mesh_assert_uploadable :: proc(vertices: []VertexData, indices: []u32) {
	assert(len(vertices) > 0)
	assert(len(indices) > 0)
	assert(len(indices) % 3 == 0)
}

map_mesh_append_brush :: proc(
	vertices: ^[dynamic]VertexData,
	indices: ^[dynamic]u32,
	brush: ^MapBrush,
	base_winding_size: f32,
) -> (emitted_faces, skipped_faces: int) {
	for face_index in 0 ..< len(brush.faces) {
		face := &brush.faces[face_index]
		winding := map_mesh_clipped_face(brush, face, face_index, base_winding_size)

		if winding.count < 3 {
			skipped_faces += 1
			continue
		}

		base := len(vertices^)
		color := quake_map_face_color(face)
		for i in 0 ..< winding.count {
			append(vertices, VertexData{pos = winding.points[i], color = color})
		}

		reverse_winding := linalg.dot(linalg.cross(winding.points[1] - winding.points[0], winding.points[2] - winding.points[0]), face.normal) < 0
		for i in 1 ..< winding.count - 1 {
			append(indices, u32(base))
			if reverse_winding {
				append(indices, u32(base + i + 1))
				append(indices, u32(base + i))
			} else {
				append(indices, u32(base + i))
				append(indices, u32(base + i + 1))
			}
		}
		emitted_faces += 1
	}
	return
}

map_mesh_clipped_face :: proc(brush: ^MapBrush, face: ^MapFace, face_index: int, base_winding_size: f32) -> MapWinding {
	winding := map_mesh_base_winding(face, base_winding_size)
	for other_face_index in 0 ..< len(brush.faces) {
		if other_face_index == face_index {
			continue
		}
		map_mesh_clip_winding(&winding, &brush.faces[other_face_index])
		if winding.count == 0 {
			break
		}
	}
	return winding
}

map_mesh_base_winding :: proc(face: ^MapFace, size: f32) -> MapWinding {
	reference := Vec3{0, 1, 0}
	if math.abs(linalg.dot(face.normal, reference)) > 0.9 {
		reference = {1, 0, 0}
	}

	right := linalg.normalize0(linalg.cross(face.normal, reference))
	forward := linalg.normalize0(linalg.cross(right, face.normal))
	center := face.normal * face.d

	winding: MapWinding
	map_mesh_append_winding_point(&winding, center + right * size + forward * size)
	map_mesh_append_winding_point(&winding, center - right * size + forward * size)
	map_mesh_append_winding_point(&winding, center - right * size - forward * size)
	map_mesh_append_winding_point(&winding, center + right * size - forward * size)
	return winding
}

map_mesh_clip_winding :: proc(winding: ^MapWinding, plane: ^MapFace) {
	if winding.count == 0 {
		return
	}

	clipped: MapWinding
	previous := winding.points[winding.count - 1]
	previous_dist := linalg.dot(plane.normal, previous) - plane.d
	previous_inside := previous_dist <= MAP_PLANE_EPSILON

	for i in 0 ..< winding.count {
		current := winding.points[i]
		current_dist := linalg.dot(plane.normal, current) - plane.d
		current_inside := current_dist <= MAP_PLANE_EPSILON

		if current_inside != previous_inside {
			t := previous_dist / (previous_dist - current_dist)
			map_mesh_append_winding_point(&clipped, previous + (current - previous) * t)
		}
		if current_inside {
			map_mesh_append_winding_point(&clipped, current)
		}

		previous = current
		previous_dist = current_dist
		previous_inside = current_inside
	}
	winding^ = clipped
}

map_mesh_append_winding_point :: proc(winding: ^MapWinding, point: Vec3) {
	assert(winding.count < MAP_MAX_WINDING_POINTS)
	if winding.count > 0 && linalg.length(winding.points[winding.count - 1] - point) < MAP_VERTEX_EPSILON {
		return
	}
	winding.points[winding.count] = point
	winding.count += 1
}

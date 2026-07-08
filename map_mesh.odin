package main

import "base:runtime"
import "core:log"
import "core:math"
import "core:math/linalg"

MAP_PLANE_EPSILON :: f32(0.001)
MAP_VERTEX_EPSILON :: f32(0.01)
MAP_HYPERPLANE_SIZE :: f32(512.0)

create_map_mesh :: proc(qmap: ^QuakeMap, allocator := context.allocator) -> CpuMesh {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	vertices := make([dynamic]VertexData, 0, 8192, allocator)
	indices := make([dynamic]u16, 0, 24576, allocator)
	skipped_faces := 0

	for &entity in qmap.entities {
		classname, ok := map_entity_property(&entity, "classname")
		if !ok || classname != "worldspawn" {
			continue
		}

		for &brush in entity.brushes {
			brush_skipped := map_mesh_append_brush(&vertices, &indices, &brush)
			skipped_faces += brush_skipped
		}
	}

	log.debugf("Built map mesh: vertices=%d indices=%d skipped_faces=%d", len(vertices), len(indices), skipped_faces)
	return CpuMesh{vertices = vertices[:], indices = indices[:], allocator = allocator}
}

map_mesh_append_brush :: proc(
	vertices: ^[dynamic]VertexData,
	indices: ^[dynamic]u16,
	brush: ^MapBrush,
) -> (skipped_faces: int) {
	for face_index in 0 ..< len(brush.faces) {
		face := &brush.faces[face_index]
		points := map_mesh_clipped_face(brush, face, face_index)

		if len(points) < 3 {
			skipped_faces += 1
			continue
		}

		base := len(vertices^)
		color := quake_map_face_color(face)
		for point in points {
			append(vertices, VertexData{pos = point, color = color})
		}

		assert(base + len(points) <= 65535)
		reverse_winding := linalg.dot(linalg.cross(points[1] - points[0], points[2] - points[0]), face.normal) < 0
		for i in 1 ..< len(points) - 1 {
			append(indices, u16(base))
			if reverse_winding {
				append(indices, u16(base + i + 1))
				append(indices, u16(base + i))
			} else {
				append(indices, u16(base + i))
				append(indices, u16(base + i + 1))
			}
		}
	}
	return
}

map_mesh_clipped_face :: proc(brush: ^MapBrush, face: ^MapFace, face_index: int) -> [dynamic]Vec3 {
	points := map_mesh_base_winding(face)
	for other_face_index in 0 ..< len(brush.faces) {
		if other_face_index == face_index {
			continue
		}
		map_mesh_clip_winding(&points, &brush.faces[other_face_index])
		if len(points) == 0 {
			break
		}
	}
	return points
}

map_mesh_base_winding :: proc(face: ^MapFace) -> [dynamic]Vec3 {
	reference := Vec3{0, 1, 0}
	if math.abs(linalg.dot(face.normal, reference)) > 0.9 {
		reference = {1, 0, 0}
	}

	right := linalg.normalize0(linalg.cross(face.normal, reference))
	forward := linalg.normalize0(linalg.cross(right, face.normal))
	center := face.normal * face.d
	size := MAP_HYPERPLANE_SIZE

	points := make([dynamic]Vec3, 0, 8, context.temp_allocator)
	append(&points, center + right * size + forward * size)
	append(&points, center - right * size + forward * size)
	append(&points, center - right * size - forward * size)
	append(&points, center + right * size - forward * size)
	return points
}

map_mesh_clip_winding :: proc(points: ^[dynamic]Vec3, plane: ^MapFace) {
	if len(points^) == 0 {
		return
	}

	clipped := make([dynamic]Vec3, 0, len(points^) + 4, context.temp_allocator)
	previous := points^[len(points^) - 1]
	previous_dist := linalg.dot(plane.normal, previous) - plane.d
	previous_inside := previous_dist <= MAP_PLANE_EPSILON

	for current in points^ {
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
	points^ = clipped
}

map_mesh_append_winding_point :: proc(points: ^[dynamic]Vec3, point: Vec3) {
	if len(points^) > 0 && linalg.length(points^[len(points^) - 1] - point) < MAP_VERTEX_EPSILON {
		return
	}
	append(points, point)
}

package main

import "core:c"
import "core:math"
import b3 "vendor:box3d"

SUZANNE_HULL_MAX_VERTICES :: 48
SUZANNE_HULL_SAMPLE_COUNT :: 48

collision_create_suzanne_hull :: proc(mesh: ^CpuMesh, source: CollisionSource) -> ^b3.HullData {
	points := collision_hull_points(mesh, source)
	hull := b3.CreateHull(raw_data(points), c.int(len(points)), SUZANNE_HULL_MAX_VERTICES)
	assert(hull != nil)
	return hull
}

collision_hull_points :: proc(mesh: ^CpuMesh, source: CollisionSource) -> []b3.Vec3 {
	#partial switch source {
	case .Authored:
		return collision_mesh_vertices(mesh)
	case .Generated_From_Visual:
		return collision_sample_support_points(mesh, SUZANNE_HULL_SAMPLE_COUNT)
	}
	return collision_sample_support_points(mesh, SUZANNE_HULL_SAMPLE_COUNT)
}

collision_mesh_vertices :: proc(mesh: ^CpuMesh) -> []b3.Vec3 {
	points := make([]b3.Vec3, len(mesh.vertices), context.temp_allocator)
	for vertex, i in mesh.vertices {
		points[i] = {vertex.pos.x, vertex.pos.y, vertex.pos.z}
	}
	return points
}

collision_sample_support_points :: proc(mesh: ^CpuMesh, sample_count: int) -> []b3.Vec3 {
	points := make([]b3.Vec3, sample_count, context.temp_allocator)
	point_count := 0
	golden_angle := f32(math.PI) * (3.0 - math.sqrt(f32(5.0)))

	for i in 0..<sample_count {
		y := 1.0 - 2.0 * (f32(i) + 0.5) / f32(sample_count)
		radius := math.sqrt(max(f32(0), 1.0 - y * y))
		angle := golden_angle * f32(i)
		direction := Vec3{math.cos(angle) * radius, y, math.sin(angle) * radius}
		point := collision_support_point(mesh, direction)
		collision_append_unique_point(points, &point_count, point)
	}

	return points[:point_count]
}

collision_support_point :: proc(mesh: ^CpuMesh, direction: Vec3) -> b3.Vec3 {
	best := mesh.vertices[0].pos
	best_dot := best.x * direction.x + best.y * direction.y + best.z * direction.z
	for vertex in mesh.vertices[1:] {
		dot := vertex.pos.x * direction.x + vertex.pos.y * direction.y + vertex.pos.z * direction.z
		if dot > best_dot {
			best = vertex.pos
			best_dot = dot
		}
	}
	return {best.x, best.y, best.z}
}

collision_append_unique_point :: proc(points: []b3.Vec3, point_count: ^int, point: b3.Vec3) {
	EPSILON_SQUARED :: f32(0.000001)
	for existing in points[:point_count^] {
		dx := existing.x - point.x
		dy := existing.y - point.y
		dz := existing.z - point.z
		if dx * dx + dy * dy + dz * dz < EPSILON_SQUARED {
			return
		}
	}
	points[point_count^] = point
	point_count^ += 1
}

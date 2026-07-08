package engine

import "base:runtime"
import "core:c"
import b3 "vendor:box3d"

physics_create_convex_hull_from_mesh :: proc(mesh: ^CpuMesh, max_vertices: int) -> ^b3.HullData {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	points := collision_mesh_unique_vertices(mesh)
	assert(len(points) <= max_vertices)
	hull := b3.CreateHull(raw_data(points), c.int(len(points)), c.int(max_vertices))
	assert(hull != nil)
	return hull
}

collision_mesh_unique_vertices :: proc(mesh: ^CpuMesh) -> []b3.Vec3 {
	points := make([]b3.Vec3, len(mesh.vertices), context.temp_allocator)
	point_count := 0
	for vertex in mesh.vertices {
		point := b3.Vec3{vertex.pos.x, vertex.pos.y, vertex.pos.z}
		collision_append_unique_point(points, &point_count, point)
	}
	return points[:point_count]
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

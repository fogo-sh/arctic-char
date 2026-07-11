package game

import engine "../engine"
import b3 "vendor:box3d"

PROP_HULL_MAX_VERTICES :: 48

collision_create_prop_hull :: proc(mesh: ^CpuMesh) -> ^b3.HullData {
	return engine.physics_create_convex_hull_from_mesh(mesh, PROP_HULL_MAX_VERTICES)
}

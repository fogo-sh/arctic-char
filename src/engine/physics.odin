package engine

import "base:runtime"
import "core:c"
import b3 "vendor:box3d"

PhysicsWorld :: struct {
	id: b3.WorldId,
}

PHYSICS_STEP_TIME :: f32(1.0 / 60.0)
PHYSICS_SUBSTEPS :: 4

physics_create :: proc() -> PhysicsWorld {
	world_def := b3.DefaultWorldDef()
	world_def.gravity = {0, -10, 0}
	return PhysicsWorld{id = b3.CreateWorld(world_def)}
}

physics_destroy :: proc(physics: ^PhysicsWorld) {
	if b3.IS_NON_NULL(physics.id) {
		b3.DestroyWorld(physics.id)
	}
	physics^ = {}
}

physics_step :: proc(physics: ^PhysicsWorld) {
	b3.World_Step(physics.id, PHYSICS_STEP_TIME, PHYSICS_SUBSTEPS)
}

physics_create_static_mesh_data :: proc(mesh: ^CpuMesh) -> ^b3.MeshData {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	assert(len(mesh.vertices) >= 3)
	assert(len(mesh.indices) >= 3 && len(mesh.indices) % 3 == 0)

	positions := make([]b3.Vec3, len(mesh.vertices), context.temp_allocator)
	for vertex, i in mesh.vertices {
		positions[i] = vertex.pos
	}

	indices := make([]i32, len(mesh.indices), context.temp_allocator)
	for index, i in mesh.indices {
		assert(index <= u32(max(i32)))
		indices[i] = i32(index)
	}

	degenerate_indices := make([]c.int, len(mesh.indices) / 3, context.temp_allocator)
	mesh_def := b3.MeshDef{
		vertices = raw_data(positions),
		indices = raw_data(indices),
		vertexCount = c.int(len(positions)),
		triangleCount = c.int(len(indices) / 3),
		weldTolerance = 0.01,
		weldVertices = true,
		identifyEdges = true,
	}
	mesh_data := b3.CreateMesh(mesh_def, raw_data(degenerate_indices), c.int(len(degenerate_indices)))
	assert(mesh_data != nil)
	return mesh_data
}

physics_body_matrix :: proc(body: b3.BodyId) -> matrix[4, 4]f32 {
	transform := b3.Body_GetTransform(body)
	rotation := b3.MakeMatrixFromQuat(transform.q)

	model: matrix[4, 4]f32
	model[0][0] = 1
	model[1][1] = 1
	model[2][2] = 1
	model[3][3] = 1
	model[0][0] = rotation[0][0]
	model[0][1] = rotation[0][1]
	model[0][2] = rotation[0][2]
	model[1][0] = rotation[1][0]
	model[1][1] = rotation[1][1]
	model[1][2] = rotation[1][2]
	model[2][0] = rotation[2][0]
	model[2][1] = rotation[2][1]
	model[2][2] = rotation[2][2]
	model[3][0] = f32(transform.p.x)
	model[3][1] = f32(transform.p.y)
	model[3][2] = f32(transform.p.z)
	return model
}

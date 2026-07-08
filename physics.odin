package main

import "base:runtime"
import "core:c"
import b3 "vendor:box3d"

PhysicsWorld :: struct {
	id:            b3.WorldId,
	suzanne_hull:  ^b3.HullData,
	map_mesh:      ^b3.MeshData,
	map_body:      b3.BodyId,
}

PHYSICS_STEP_TIME :: f32(1.0 / 60.0)
PHYSICS_SUBSTEPS :: 4

COLLISION_CATEGORY_WORLD :: u64(1 << 0)
COLLISION_CATEGORY_PROP :: u64(1 << 1)
COLLISION_CATEGORY_PLAYER :: u64(1 << 2)
COLLISION_MASK_ALL :: u64(COLLISION_CATEGORY_WORLD | COLLISION_CATEGORY_PROP | COLLISION_CATEGORY_PLAYER)

physics_create :: proc(collision_mesh: ^CpuMesh, level_mesh: ^CpuMesh) -> PhysicsWorld {
	world_def := b3.DefaultWorldDef()
	world_def.gravity = {0, -10, 0}

	physics := PhysicsWorld{
		id = b3.CreateWorld(world_def),
		suzanne_hull = collision_create_suzanne_hull(collision_mesh),
	}
	physics.map_mesh = physics_create_static_mesh_data(level_mesh)
	physics.map_body = physics_create_map_body(&physics)
	return physics
}

physics_destroy :: proc(physics: ^PhysicsWorld) {
	if b3.IS_NON_NULL(physics.id) {
		b3.DestroyWorld(physics.id)
	}
	if physics.map_mesh != nil {
		b3.DestroyMesh(physics.map_mesh)
	}
	if physics.suzanne_hull != nil {
		b3.DestroyHull(physics.suzanne_hull)
	}
	physics^ = {}
}

physics_step :: proc(physics: ^PhysicsWorld) {
	b3.World_Step(physics.id, PHYSICS_STEP_TIME, PHYSICS_SUBSTEPS)
}

physics_player_query_filter :: proc() -> b3.QueryFilter {
	return {categoryBits = COLLISION_CATEGORY_PLAYER, maskBits = COLLISION_CATEGORY_WORLD | COLLISION_CATEGORY_PROP}
}

physics_world_shape_filter :: proc() -> b3.Filter {
	return {categoryBits = COLLISION_CATEGORY_WORLD, maskBits = COLLISION_MASK_ALL}
}

physics_prop_shape_filter :: proc() -> b3.Filter {
	return {categoryBits = COLLISION_CATEGORY_PROP, maskBits = COLLISION_MASK_ALL}
}

physics_create_map_body :: proc(physics: ^PhysicsWorld) -> b3.BodyId {
	body_def := b3.DefaultBodyDef()
	body := b3.CreateBody(physics.id, body_def)

	shape_def := b3.DefaultShapeDef()
	shape_def.baseMaterial.friction = 0.7
	shape_def.filter = physics_world_shape_filter()
	_ = b3.CreateMeshShape(body, shape_def, physics.map_mesh, {1, 1, 1})
	return body
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

physics_create_suzanne_body :: proc(physics: ^PhysicsWorld, position: Vec3, ordinal: int) -> b3.BodyId {
	body_def := b3.DefaultBodyDef()
	body_def.type = .dynamicBody
	body_def.position = {position.x, position.y, position.z}
	body_def.angularVelocity = {
		0.25 + f32(ordinal % 5) * 0.12,
		0.35 + f32(ordinal % 7) * 0.08,
		0.15 + f32(ordinal % 3) * 0.16,
	}
	body := b3.CreateBody(physics.id, body_def)

	shape_def := b3.DefaultShapeDef()
	shape_def.density = 1.0
	shape_def.baseMaterial.friction = 0.6
	shape_def.baseMaterial.restitution = 0.05
	shape_def.filter = physics_prop_shape_filter()
	_ = b3.CreateHullShape(body, shape_def, physics.suzanne_hull)

	return body
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

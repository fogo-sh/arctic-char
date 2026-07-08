package main

import b3 "vendor:box3d"

PhysicsWorld :: struct {
	id:            b3.WorldId,
	suzanne_hull:  ^b3.HullData,
}

PHYSICS_STEP_TIME :: f32(1.0 / 60.0)
PHYSICS_SUBSTEPS :: 4

COLLISION_CATEGORY_WORLD :: u64(1 << 0)
COLLISION_CATEGORY_PROP :: u64(1 << 1)
COLLISION_CATEGORY_PLAYER :: u64(1 << 2)
COLLISION_MASK_ALL :: u64(COLLISION_CATEGORY_WORLD | COLLISION_CATEGORY_PROP | COLLISION_CATEGORY_PLAYER)

physics_create :: proc(collision_mesh: ^CpuMesh) -> PhysicsWorld {
	world_def := b3.DefaultWorldDef()
	world_def.gravity = {0, -10, 0}

	physics := PhysicsWorld{
		id = b3.CreateWorld(world_def),
		suzanne_hull = collision_create_suzanne_hull(collision_mesh),
	}
	return physics
}

physics_destroy :: proc(physics: ^PhysicsWorld) {
	if b3.IS_NON_NULL(physics.id) {
		b3.DestroyWorld(physics.id)
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

physics_create_ground_body :: proc(physics: ^PhysicsWorld) -> b3.BodyId {
	body_def := b3.DefaultBodyDef()
	body_def.position = {0, -0.5, 0}
	body := b3.CreateBody(physics.id, body_def)

	shape_def := b3.DefaultShapeDef()
	shape_def.baseMaterial.friction = 0.7
	shape_def.filter = physics_world_shape_filter()
	ground_hull := b3.MakeBoxHull(50, 0.5, 50)
	_ = b3.CreateHullShape(body, shape_def, &ground_hull.base)
	return body
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

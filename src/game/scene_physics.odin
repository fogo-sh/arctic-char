package game

import engine "../engine"
import b3 "vendor:box3d"

COLLISION_CATEGORY_WORLD :: u64(1 << 0)
COLLISION_CATEGORY_PROP :: u64(1 << 1)
COLLISION_CATEGORY_PLAYER :: u64(1 << 2)
COLLISION_MASK_ALL :: u64(COLLISION_CATEGORY_WORLD | COLLISION_CATEGORY_PROP | COLLISION_CATEGORY_PLAYER)

ScenePhysicsAssets :: struct {
	suzanne_hull: ^b3.HullData,
	map_mesh:     ^b3.MeshData,
	map_body:     b3.BodyId,
}

scene_physics_assets_create :: proc(scene: ^Scene, collision_mesh, level_mesh: ^CpuMesh) {
	scene.physics_assets.suzanne_hull = collision_create_suzanne_hull(collision_mesh)
	scene.physics_assets.map_mesh = engine.physics_create_static_mesh_data(level_mesh)
	scene.physics_assets.map_body = scene_physics_create_map_body(scene)
}

scene_physics_assets_destroy :: proc(scene: ^Scene) {
	if scene.physics_assets.map_mesh != nil {
		b3.DestroyMesh(scene.physics_assets.map_mesh)
	}
	if scene.physics_assets.suzanne_hull != nil {
		b3.DestroyHull(scene.physics_assets.suzanne_hull)
	}
	scene.physics_assets = {}
}

scene_physics_replace_map_mesh :: proc(scene: ^Scene, level_mesh: ^CpuMesh) {
	if b3.IS_NON_NULL(scene.physics_assets.map_body) {
		b3.DestroyBody(scene.physics_assets.map_body)
	}
	if scene.physics_assets.map_mesh != nil {
		b3.DestroyMesh(scene.physics_assets.map_mesh)
	}
	scene.physics_assets.map_mesh = engine.physics_create_static_mesh_data(level_mesh)
	scene.physics_assets.map_body = scene_physics_create_map_body(scene)
}

scene_physics_create_map_body :: proc(scene: ^Scene) -> b3.BodyId {
	body_def := b3.DefaultBodyDef()
	body := b3.CreateBody(scene.physics.id, body_def)

	shape_def := b3.DefaultShapeDef()
	shape_def.baseMaterial.friction = 0.7
	shape_def.filter = scene_physics_world_shape_filter()
	_ = b3.CreateMeshShape(body, shape_def, scene.physics_assets.map_mesh, {1, 1, 1})
	return body
}

scene_physics_create_suzanne_body :: proc(scene: ^Scene, position: Vec3, ordinal: int) -> b3.BodyId {
	body_def := b3.DefaultBodyDef()
	body_def.type = .dynamicBody
	body_def.position = {position.x, position.y, position.z}
	body_def.angularVelocity = scene_physics_suzanne_angular_velocity(ordinal)
	body := b3.CreateBody(scene.physics.id, body_def)

	shape_def := b3.DefaultShapeDef()
	shape_def.density = 1.0
	shape_def.baseMaterial.friction = 0.6
	shape_def.baseMaterial.restitution = 0.05
	shape_def.filter = scene_physics_prop_shape_filter()
	_ = b3.CreateHullShape(body, shape_def, scene.physics_assets.suzanne_hull)
	return body
}

scene_physics_suzanne_angular_velocity :: proc(ordinal: int) -> Vec3 {
	return {
		0.25 + f32(ordinal % 5) * 0.12,
		0.35 + f32(ordinal % 7) * 0.08,
		0.15 + f32(ordinal % 3) * 0.16,
	}
}

physics_player_query_filter :: proc() -> b3.QueryFilter {
	return {categoryBits = COLLISION_CATEGORY_PLAYER, maskBits = COLLISION_CATEGORY_WORLD | COLLISION_CATEGORY_PROP}
}

scene_physics_world_shape_filter :: proc() -> b3.Filter {
	return {categoryBits = COLLISION_CATEGORY_WORLD, maskBits = COLLISION_MASK_ALL}
}

scene_physics_prop_shape_filter :: proc() -> b3.Filter {
	return {categoryBits = COLLISION_CATEGORY_PROP, maskBits = COLLISION_MASK_ALL}
}

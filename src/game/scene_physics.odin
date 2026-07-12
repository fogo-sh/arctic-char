package game

import engine "../engine"
import "core:math/linalg"
import b3 "vendor:box3d"

COLLISION_CATEGORY_WORLD :: u64(1 << 0)
COLLISION_CATEGORY_PROP :: u64(1 << 1)
COLLISION_CATEGORY_PLAYER :: u64(1 << 2)
COLLISION_CATEGORY_TRIGGER :: u64(1 << 3)
COLLISION_MASK_ALL :: u64(COLLISION_CATEGORY_WORLD | COLLISION_CATEGORY_PROP | COLLISION_CATEGORY_PLAYER | COLLISION_CATEGORY_TRIGGER)

ScenePhysicsAssets :: struct {
	prop_hulls: [MAX_PROP_ASSETS]^b3.HullData,
	prop_count: int,
	map_mesh:   ^b3.MeshData,
	map_body:   b3.BodyId,
}

scene_physics_assets_create :: proc(scene: ^Scene, assets: ^LoadedSceneAssets) {
	scene.physics_assets.prop_count = len(assets.prop_assets)
	for &asset, i in assets.prop_assets {
		scene.physics_assets.prop_hulls[i] = collision_create_prop_hull(&asset.collision_mesh)
	}
	scene.physics_assets.map_mesh = engine.physics_create_static_mesh_data(&assets.level.render_mesh)
	scene.physics_assets.map_body = scene_physics_create_map_body(scene)
}

scene_physics_assets_destroy :: proc(scene: ^Scene) {
	if b3.IS_NON_NULL(scene.physics_assets.map_body) {
		b3.DestroyBody(scene.physics_assets.map_body)
	}
	if scene.physics_assets.map_mesh != nil {
		b3.DestroyMesh(scene.physics_assets.map_mesh)
	}
	for hull in scene.physics_assets.prop_hulls {
		if hull != nil {
			b3.DestroyHull(hull)
		}
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

scene_physics_create_prop_body :: proc(scene: ^Scene, position: Vec3, prop_asset_index: u16, ordinal: int) -> b3.BodyId {
	return scene_physics_create_prop_body_with_type(scene, position, linalg.QUATERNIONF32_IDENTITY, prop_asset_index, ordinal, .dynamicBody)
}

scene_physics_create_prop_proxy_body :: proc(scene: ^Scene, position: Vec3, rotation: linalg.Quaternionf32, prop_asset_index: u16) -> b3.BodyId {
	return scene_physics_create_prop_body_with_type(scene, position, rotation, prop_asset_index, 0, .kinematicBody)
}

scene_physics_create_prop_body_with_type :: proc(scene: ^Scene, position: Vec3, rotation: linalg.Quaternionf32, prop_asset_index: u16, ordinal: int, body_type: b3.BodyType) -> b3.BodyId {
	body_def := b3.DefaultBodyDef()
	body_def.type = body_type
	body_def.position = {position.x, position.y, position.z}
	body_def.rotation = b3.Quat(rotation)
	if body_type == .dynamicBody {
		body_def.angularVelocity = scene_physics_prop_angular_velocity(ordinal)
	}
	body := b3.CreateBody(scene.physics.id, body_def)

	shape_def := b3.DefaultShapeDef()
	shape_def.density = 1.0
	shape_def.baseMaterial.friction = 0.6
	shape_def.baseMaterial.restitution = 0.05
	shape_def.filter = scene_physics_prop_shape_filter()
	index := int(prop_asset_index)
	if index < 0 || index >= scene.physics_assets.prop_count {
		index = 0
	}
	_ = b3.CreateHullShape(body, shape_def, scene.physics_assets.prop_hulls[index])
	return body
}

scene_physics_create_trigger_body :: proc(scene: ^Scene, center, half_extents: Vec3) -> (body: b3.BodyId, shape: b3.ShapeId) {
	body_def := b3.DefaultBodyDef()
	body_def.position = b3.Pos(center)
	body = b3.CreateBody(scene.physics.id, body_def)

	shape_def := b3.DefaultShapeDef()
	shape_def.isSensor = true
	shape_def.filter = scene_physics_trigger_shape_filter()
	box := b3.MakeBoxHull(half_extents.x, half_extents.y, half_extents.z)
	shape = b3.CreateHullShape(body, shape_def, &box.base)
	return
}

scene_physics_prop_angular_velocity :: proc(ordinal: int) -> Vec3 {
	return {
		0.25 + f32(ordinal % 5) * 0.12,
		0.35 + f32(ordinal % 7) * 0.08,
		0.15 + f32(ordinal % 3) * 0.16,
	}
}

physics_player_query_filter :: proc() -> b3.QueryFilter {
	return {categoryBits = COLLISION_CATEGORY_PLAYER, maskBits = COLLISION_CATEGORY_WORLD | COLLISION_CATEGORY_PROP}
}

physics_player_trigger_query_filter :: proc() -> b3.QueryFilter {
	return {categoryBits = COLLISION_CATEGORY_PLAYER, maskBits = COLLISION_CATEGORY_TRIGGER}
}

scene_physics_world_shape_filter :: proc() -> b3.Filter {
	return {categoryBits = COLLISION_CATEGORY_WORLD, maskBits = COLLISION_MASK_ALL}
}

scene_physics_prop_shape_filter :: proc() -> b3.Filter {
	return {categoryBits = COLLISION_CATEGORY_PROP, maskBits = COLLISION_MASK_ALL}
}

scene_physics_trigger_shape_filter :: proc() -> b3.Filter {
	return {categoryBits = COLLISION_CATEGORY_TRIGGER, maskBits = COLLISION_CATEGORY_PLAYER}
}

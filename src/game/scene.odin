package game

import "base:runtime"
import "core:math"
import "core:math/linalg"
import engine "../engine"
import b3 "vendor:box3d"

MAX_SUZANNES :: 100
SPAWN_INTERVAL :: f32(0.12)

Scene :: struct {
	allocator:      runtime.Allocator,
	physics:        PhysicsWorld,
	physics_assets: ScenePhysicsAssets,
	player:         PlayerController,
	objects:        [dynamic]Object,
	suzanne_mesh:   MeshHandle,
	map_mesh:       MeshHandle,
	next_object_id: ObjectId,
	spawn_timer:    f32,
	spawned_count:  int,
	accumulator:    f32,
}

ObjectKind :: enum {
	Map,
	Suzanne,
}

ObjectId :: distinct u32

Transform :: struct {
	position: Vec3,
}

RenderObject :: struct {
	mesh:    MeshHandle,
	visible: bool,
}

PhysicsObject :: struct {
	body:           b3.BodyId,
	enabled:        bool,
	sync_transform: bool,
}

Object :: struct {
	id:        ObjectId,
	name:      string,
	kind:      ObjectKind,
	transform: Transform,
	render:    RenderObject,
	physics:   PhysicsObject,
}

scene_create :: proc(assets: ^LoadedSceneAssets, gpu: SceneGpuResources, allocator := context.allocator) -> Scene {
	scene := Scene{
		allocator = allocator,
		physics = engine.physics_create(),
		player = player_create(assets.level.player_spawn.position, assets.level.player_spawn.yaw),
		objects = make([dynamic]Object, 0, MAX_SUZANNES + 1, allocator),
		suzanne_mesh = gpu.suzanne_handle,
		map_mesh = gpu.map_handle,
		next_object_id = 1,
		spawn_timer = SPAWN_INTERVAL,
	}
	scene_physics_assets_create(&scene, &assets.collision_mesh, &assets.level.render_mesh)
	scene_create_map(&scene)
	return scene
}

scene_destroy :: proc(scene: ^Scene) {
	delete(scene.objects)
	engine.physics_destroy(&scene.physics)
	scene_physics_assets_destroy(scene)
	scene^ = {}
}

scene_reload_level :: proc(scene: ^Scene, level: ^LevelAsset) {
	scene_physics_replace_map_mesh(scene, &level.render_mesh)
	scene.player = player_create(level.player_spawn.position, level.player_spawn.yaw)
	scene.accumulator = 0
}

scene_update :: proc(scene: ^Scene, move_input: PlayerMoveInput, look_input: PlayerLookInput, delta_time: f32) {
	player_apply_look(&scene.player, look_input)

	scene.accumulator += min(delta_time, 0.25)
	for scene.accumulator >= PHYSICS_STEP_TIME {
		scene_fixed_update(scene, move_input, PHYSICS_STEP_TIME)
		scene.accumulator -= PHYSICS_STEP_TIME
	}
}

scene_fixed_update :: proc(scene: ^Scene, input: PlayerMoveInput, step_time: f32) {
	scene.spawn_timer += step_time
	for scene.spawned_count < MAX_SUZANNES && scene.spawn_timer >= SPAWN_INTERVAL {
		scene_spawn_suzanne(scene)
		scene.spawn_timer -= SPAWN_INTERVAL
	}

	player_update(&scene.player, &scene.physics, input, step_time)

	engine.physics_step(&scene.physics)
}

scene_create_map :: proc(scene: ^Scene) {
	scene_add_object(scene, Object{
		name = "Map",
		kind = .Map,
		transform = {position = {0, 0, 0}},
		render = {mesh = scene.map_mesh, visible = true},
		physics = {enabled = false},
	})
}

scene_spawn_suzanne :: proc(scene: ^Scene) {
	i := scene.spawned_count
	angle := f32(i) * 2.3999632
	radius := 0.15 + f32(i % 8) * 0.08
	position := Vec3{
		math.cos(angle) * radius,
		7.0,
		math.sin(angle) * radius,
	}

	body := scene_physics_create_suzanne_body(scene, position, i)
	scene_add_object(scene, Object{
		name = "Suzanne",
		kind = .Suzanne,
		transform = {position = position},
		render = {mesh = scene.suzanne_mesh, visible = true},
		physics = {body = body, enabled = true, sync_transform = true},
	})
	scene.spawned_count += 1
}

scene_add_object :: proc(scene: ^Scene, object: Object) -> ObjectId {
	id := scene.next_object_id
	scene.next_object_id = ObjectId(u32(scene.next_object_id) + 1)
	stored := object
	stored.id = id
	append(&scene.objects, stored)
	return id
}

scene_collect_render_items :: proc(scene: ^Scene, items: ^[dynamic]RenderItem) -> []RenderItem {
	clear(items)
	for &object in scene.objects {
		if !object.render.visible {
			continue
		}
		model := scene_object_model_matrix(&object)
		append(items, RenderItem{mesh = object.render.mesh, model = model})
	}
	return items[:]
}

scene_render_globals :: proc(scene: ^Scene, window_size: [2]i32) -> RenderPassGlobals {
	aspect := f32(window_size.x) / f32(window_size.y)
	return RenderPassGlobals{
		view = player_view_matrix(&scene.player),
		proj = linalg.matrix4_perspective_f32(linalg.to_radians(f32(70)), aspect, 0.1, 100),
	}
}

scene_object_model_matrix :: proc(object: ^Object) -> matrix[4, 4]f32 {
	if object.physics.sync_transform {
		return engine.physics_body_matrix(object.physics.body)
	}
	return linalg.matrix4_translate_f32(object.transform.position)
}

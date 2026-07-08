package main

import "base:runtime"
import "core:math"
import "core:math/linalg"
import b3 "vendor:box3d"

MAX_SUZANNES :: 100
SPAWN_INTERVAL :: f32(0.12)

Scene :: struct {
	allocator:    runtime.Allocator,
	physics:      PhysicsWorld,
	player_spawn: MapPlayerSpawn,
	player_id:    ObjectId,
	objects:      [dynamic]Object,
	spawn_timer:  f32,
	spawned_count: int,
	accumulator:  f32,
}

ObjectKind :: enum {
	Ground,
	Map,
	Player,
	Suzanne,
}

ObjectId :: distinct int

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
	player:    PlayerController,
}

scene_create :: proc(assets: ^SceneAssets, allocator := context.allocator) -> Scene {
	player_spawn := MapPlayerSpawn{position = PLAYER_SPEC.spawn_position, yaw = 0}
	if map_spawn, ok := quake_map_find_player_spawn(&assets.level_map); ok {
		player_spawn = map_spawn
	}

	scene := Scene{
		allocator = allocator,
		physics = physics_create(&assets.collision_mesh),
		player_spawn = player_spawn,
		objects = make([dynamic]Object, 0, MAX_SUZANNES + 3, allocator),
		spawn_timer = SPAWN_INTERVAL,
	}
	scene_create_ground(&scene)
	scene_create_map(&scene)
	scene_create_player(&scene)
	return scene
}

scene_destroy :: proc(scene: ^Scene) {
	delete(scene.objects)
	physics_destroy(&scene.physics)
	scene^ = {}
}

scene_update :: proc(scene: ^Scene, input: PlayerInput, delta_time: f32) {
	scene.accumulator += min(delta_time, 0.25)
	for scene.accumulator >= PHYSICS_STEP_TIME {
		scene_fixed_update(scene, input, PHYSICS_STEP_TIME)
		scene.accumulator -= PHYSICS_STEP_TIME
	}
}

scene_fixed_update :: proc(scene: ^Scene, input: PlayerInput, step_time: f32) {
	scene.spawn_timer += step_time
	for scene.spawned_count < MAX_SUZANNES && scene.spawn_timer >= SPAWN_INTERVAL {
		scene_spawn_suzanne(scene)
		scene.spawn_timer -= SPAWN_INTERVAL
	}

	player_object := scene_player_object(scene)
	player_update(&player_object.player, &scene.physics, input, step_time)
	player_object.transform.position = player_object.player.position

	physics_step(&scene.physics)
}

scene_create_ground :: proc(scene: ^Scene) {
	body := physics_create_ground_body(&scene.physics)
	append(&scene.objects, Object{
		id = ObjectId(len(scene.objects)),
		name = "Ground",
		kind = .Ground,
		transform = {position = {0, 0, 0}},
		render = {mesh = SCENE_MESH_GROUND, visible = true},
		physics = {body = body, enabled = true, sync_transform = false},
	})
}

scene_create_map :: proc(scene: ^Scene) {
	append(&scene.objects, Object{
		id = ObjectId(len(scene.objects)),
		name = "Map",
		kind = .Map,
		transform = {position = {0, 0, 0}},
		render = {mesh = SCENE_MESH_MAP, visible = true},
		physics = {enabled = false},
	})
}

scene_create_player :: proc(scene: ^Scene) {
	player := player_create(scene.player_spawn.position, scene.player_spawn.yaw)
	scene.player_id = ObjectId(len(scene.objects))
	append(&scene.objects, Object{
		id = scene.player_id,
		name = "Player",
		kind = .Player,
		transform = {position = player.position},
		render = {visible = false},
		physics = {enabled = false},
		player = player,
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

	body := physics_create_suzanne_body(&scene.physics, position, i)
	append(&scene.objects, Object{
		id = ObjectId(len(scene.objects)),
		name = "Suzanne",
		kind = .Suzanne,
		transform = {position = position},
		render = {mesh = SCENE_MESH_SUZANNE, visible = true},
		physics = {body = body, enabled = true, sync_transform = true},
	})
	scene.spawned_count += 1
}

scene_collect_render_items :: proc(scene: ^Scene, window_size: [2]i32, items: ^[dynamic]RenderItem) -> []RenderItem {
	clear(items)
	aspect := f32(window_size.x) / f32(window_size.y)
	view := player_view_matrix(&scene_player_object(scene).player)
	proj := linalg.matrix4_perspective_f32(linalg.to_radians(f32(70)), aspect, 0.1, 100)

	for &object in scene.objects {
		if !object.render.visible {
			continue
		}
		model := scene_object_model_matrix(&object)
		append(items, RenderItem{mesh = object.render.mesh, mvp = proj * view * model})
	}
	return items[:]
}

scene_player_object :: proc(scene: ^Scene) -> ^Object {
	index := int(scene.player_id)
	assert(0 <= index && index < len(scene.objects))
	return &scene.objects[index]
}

scene_object_model_matrix :: proc(object: ^Object) -> matrix[4, 4]f32 {
	if object.physics.sync_transform {
		return physics_body_matrix(object.physics.body)
	}
	return linalg.matrix4_translate_f32(object.transform.position)
}

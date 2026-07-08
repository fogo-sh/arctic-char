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
	objects:      [dynamic]Object,
	spawn_timer:  f32,
	spawned_count: int,
}

ObjectKind :: enum {
	Ground,
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
}

scene_create :: proc(assets: ^SceneAssets, allocator := context.allocator) -> Scene {
	scene := Scene{
		allocator = allocator,
		physics = physics_create(&assets.collision_mesh),
		objects = make([dynamic]Object, 0, MAX_SUZANNES + 1, allocator),
		spawn_timer = SPAWN_INTERVAL,
	}
	scene_create_ground(&scene)
	return scene
}

scene_destroy :: proc(scene: ^Scene) {
	delete(scene.objects)
	physics_destroy(&scene.physics)
	scene^ = {}
}

scene_update :: proc(scene: ^Scene, delta_time: f32) {
	scene.spawn_timer += delta_time
	for scene.spawned_count < MAX_SUZANNES && scene.spawn_timer >= SPAWN_INTERVAL {
		scene_spawn_suzanne(scene)
		scene.spawn_timer -= SPAWN_INTERVAL
	}

	physics_step(&scene.physics, delta_time)
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
	view := linalg.matrix4_translate_f32({0, -2.0, -13.0})
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

scene_object_model_matrix :: proc(object: ^Object) -> matrix[4, 4]f32 {
	if object.physics.sync_transform {
		return physics_body_matrix(object.physics.body)
	}
	return linalg.matrix4_translate_f32(object.transform.position)
}

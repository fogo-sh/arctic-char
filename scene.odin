package main

import "core:math"
import "core:math/linalg"
import b3 "vendor:box3d"

MAX_SUZANNES :: 100
SPAWN_INTERVAL :: f32(0.12)

Scene :: struct {
	physics:      PhysicsWorld,
	objects:      [dynamic]SceneObject,
	spawn_timer:  f32,
	spawned_count: int,
}

SceneObject :: struct {
	body: b3.BodyId,
}

scene_create :: proc(assets: ^SceneAssets) -> Scene {
	scene := Scene{
		physics = physics_create(scene_assets_collision_mesh(assets), assets.collision_source),
		spawn_timer = SPAWN_INTERVAL,
	}
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
	append(&scene.objects, SceneObject{body = body})
	scene.spawned_count += 1
}

scene_collect_render_items :: proc(scene: ^Scene, window_size: [2]i32) -> []RenderItem {
	items := make([]RenderItem, len(scene.objects) + 1, context.temp_allocator)
	aspect := f32(window_size.x) / f32(window_size.y)
	view := linalg.matrix4_translate_f32({0, -2.0, -13.0})
	proj := linalg.matrix4_perspective_f32(linalg.to_radians(f32(70)), aspect, 0.1, 100)
	items[0] = RenderItem{mesh = SCENE_MESH_GROUND, mvp = proj * view}

	for object, i in scene.objects {
		items[i + 1] = RenderItem{mesh = SCENE_MESH_SUZANNE, mvp = proj * view * physics_body_matrix(object.body)}
	}
	return items
}

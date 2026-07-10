package game

import engine "../engine"
import "base:runtime"
import "core:math/linalg"
import b3 "vendor:box3d"

SPAWN_INTERVAL :: f32(0.12)
MAX_OBJECTS :: 128

Scene :: struct {
	allocator:      runtime.Allocator,
	physics:        PhysicsWorld,
	physics_assets: ScenePhysicsAssets,
	player:         PlayerController,
	objects:        [dynamic]Object,
	suzanne_mesh:   MeshHandle,
	map_mesh:       MeshHandle,
	next_object_id: ObjectId,
	accumulator:    f32,
}

ObjectKind :: enum {
	Map,
	Suzanne,
	Spawner,
	Trigger,
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
	body:             b3.BodyId,
	enabled:          bool,
	sync_transform:   bool,
	linear_velocity:  Vec3,
	angular_velocity: Vec3,
}

ThinkKind :: enum {
	None,
	SpawnerSuzanne,
}

ThinkObject :: struct {
	kind:          ThinkKind,
	interval:      f32,
	timer:         f32,
	max_count:     int,
	spawned_count: int,
}

TouchKind :: enum {
	None,
	TriggerTeleport,
}

TouchObject :: struct {
	kind:            TouchKind,
	bounds_min:      Vec3,
	bounds_max:      Vec3,
	target_position: Vec3,
	target_yaw:      f32,
}

Object :: struct {
	id:        ObjectId,
	name:      string,
	kind:      ObjectKind,
	transform: Transform,
	render:    RenderObject,
	physics:   PhysicsObject,
	think:     ThinkObject,
	touch:     TouchObject,
}

scene_create :: proc(
	assets: ^LoadedSceneAssets,
	gpu: SceneGpuResources,
	allocator := context.allocator,
) -> Scene {
	scene := Scene {
		allocator      = allocator,
		physics        = engine.physics_create(),
		player         = player_create(
			assets.level.player_spawn.position,
			assets.level.player_spawn.yaw,
		),
		objects        = make([dynamic]Object, 0, MAX_OBJECTS, allocator),
		suzanne_mesh   = gpu.suzanne_handle,
		map_mesh       = gpu.map_handle,
		next_object_id = 1,
	}
	scene_physics_assets_create(&scene, &assets.collision_mesh, &assets.level.render_mesh)
	scene_create_map(&scene)
	scene_spawn_level_entities(&scene, &assets.level.source)
	return scene
}

scene_destroy :: proc(scene: ^Scene) {
	delete(scene.objects)
	engine.physics_destroy(&scene.physics)
	scene_physics_assets_destroy(scene)
	scene^ = {}
}

scene_prepare_hot_reload :: proc(scene: ^Scene) {
	for &object in scene.objects {
		if !object.physics.enabled || !b3.IS_NON_NULL(object.physics.body) {
			continue
		}
		if object.physics.sync_transform {
			transform := b3.Body_GetTransform(object.physics.body)
			object.transform.position = Vec3(transform.p)
		}
		object.physics.linear_velocity = Vec3(b3.Body_GetLinearVelocity(object.physics.body))
		object.physics.angular_velocity = Vec3(b3.Body_GetAngularVelocity(object.physics.body))
		object.physics.body = {}
	}
	engine.physics_destroy(&scene.physics)
	scene_physics_assets_destroy(scene)
}

scene_rebuild_after_hot_reload :: proc(scene: ^Scene, fs: ^GameFS, map_qpath: string) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	level := level_load(fs, map_qpath)
	defer level_destroy(&level)
	collision_mesh := engine.load_glb_mesh(fs, "models/suzanne_collision.glb")
	defer engine.cpu_mesh_destroy(&collision_mesh)

	scene.physics = engine.physics_create()
	scene_physics_assets_create(scene, &collision_mesh, &level.render_mesh)
	for &object, index in scene.objects {
		switch object.kind {
		case .Suzanne:
			body := scene_physics_create_suzanne_body(scene, object.transform.position, int(index))
			b3.Body_SetLinearVelocity(body, object.physics.linear_velocity)
			b3.Body_SetAngularVelocity(body, object.physics.angular_velocity)
			object.physics.body = body
		case .Map:
			object.physics = {
				enabled = false,
			}
		case .Spawner, .Trigger:
			object.physics = {enabled = false}
		}
	}
	scene.accumulator = 0
}

scene_reload_level :: proc(scene: ^Scene, level: ^LevelAsset) {
	scene_physics_replace_map_mesh(scene, &level.render_mesh)
	scene.player = player_create(level.player_spawn.position, level.player_spawn.yaw)
	scene_clear_level_entities(scene)
	scene_spawn_level_entities(scene, &level.source)
	scene.accumulator = 0
}

scene_update :: proc(
	scene: ^Scene,
	move_input: PlayerMoveInput,
	look_input: PlayerLookInput,
	delta_time: f32,
) {
	player_apply_look(&scene.player, look_input)

	scene.accumulator += min(delta_time, 0.25)
	for scene.accumulator >= PHYSICS_STEP_TIME {
		scene_fixed_update(scene, move_input, PHYSICS_STEP_TIME)
		scene.accumulator -= PHYSICS_STEP_TIME
	}
}

scene_fixed_update :: proc(scene: ^Scene, input: PlayerMoveInput, step_time: f32) {
	scene_run_think(scene, step_time)

	move := player_update(&scene.player, &scene.physics, input, step_time)
	scene_touch_player(scene, move)

	engine.physics_step(&scene.physics)
}

scene_create_map :: proc(scene: ^Scene) {
	scene_add_object(
		scene,
		Object {
			name = "Map",
			kind = .Map,
			transform = {position = {0, 0, 0}},
			render = {mesh = scene.map_mesh, visible = true},
			physics = {enabled = false},
		},
	)
}

scene_spawn_suzanne :: proc(scene: ^Scene, position: Vec3) -> bool {
	i := scene_count_objects(scene, .Suzanne)
	if len(scene.objects) >= cap(scene.objects) {
		return false
	}

	body := scene_physics_create_suzanne_body(scene, position, i)
	scene_add_object(
		scene,
		Object {
			name = "Suzanne",
			kind = .Suzanne,
			transform = {position = position},
			render = {mesh = scene.suzanne_mesh, visible = true},
			physics = {
				body = body,
				enabled = true,
				sync_transform = true,
				angular_velocity = scene_physics_suzanne_angular_velocity(i),
			},
		},
	)
	return true
}

scene_add_object :: proc(scene: ^Scene, object: Object) -> ObjectId {
	assert(len(scene.objects) < cap(scene.objects))
	id := scene.next_object_id
	scene.next_object_id = ObjectId(u32(scene.next_object_id) + 1)
	stored := object
	stored.id = id
	append(&scene.objects, stored)
	return id
}

scene_count_objects :: proc(scene: ^Scene, kind: ObjectKind) -> int {
	count := 0
	for object in scene.objects {
		if object.kind == kind {
			count += 1
		}
	}
	return count
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
	return RenderPassGlobals {
		view = player_view_matrix(&scene.player),
		proj = engine.matrix4_perspective_z0_f32(linalg.to_radians(f32(70)), aspect, 0.1, 100),
	}
}

scene_object_model_matrix :: proc(object: ^Object) -> matrix[4, 4]f32 {
	if object.physics.sync_transform {
		return engine.physics_body_matrix(object.physics.body)
	}
	return linalg.matrix4_translate_f32(object.transform.position)
}

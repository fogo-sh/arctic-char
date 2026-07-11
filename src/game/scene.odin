package game

import engine "../engine"
import protocol "../protocol"
import "base:runtime"
import "core:log"
import "core:math/linalg"
import b3 "vendor:box3d"
import sdl "vendor:sdl3"

SPAWN_INTERVAL :: f32(0.12)
MAX_OBJECTS :: 2048
MAX_PLAYERS :: 32
LOCAL_PLAYER_ID :: u32(1)

Scene :: struct {
	allocator:      runtime.Allocator,
	physics:        PhysicsWorld,
	physics_assets: ScenePhysicsAssets,
	players:        [dynamic]ScenePlayer,
	camera_player_id: u32,
	player_spawn_position: Vec3,
	player_spawn_yaw: f32,
	objects:        [dynamic]Object,
	suzanne_mesh:   MeshHandle,
	map_mesh:       MeshHandle,
	next_object_id: ObjectId,
	accumulator:    f32,
	profile:        SceneProfile,
	profile_log_timer: f32,
}

ScenePlayer :: struct {
	id:         u32,
	controller: PlayerController,
}

SceneProfile :: struct {
	update_ms: f32,
	fixed_update_ms: f32,
	think_ms: f32,
	player_ms: f32,
	touch_ms: f32,
	box3d_step_ms: f32,
	fixed_steps: int,
	box3d: b3.Profile,
	counters: b3.Counters,
}

ObjectKind :: enum {
	Map,
	Suzanne,
	RemotePlayer,
	Spawner,
	Trigger,
}

ObjectId :: distinct u32

Transform :: struct {
	position: Vec3,
	yaw:      f32,
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
	net_id:    u32,
}

scene_create :: proc(
	assets: ^LoadedSceneAssets,
	gpu: SceneGpuResources,
	allocator := context.allocator,
) -> Scene {
	scene := Scene {
		allocator      = allocator,
		physics        = engine.physics_create(),
		players        = make([dynamic]ScenePlayer, 0, MAX_PLAYERS, allocator),
		camera_player_id = LOCAL_PLAYER_ID,
		player_spawn_position = assets.level.player_spawn.position,
		player_spawn_yaw = assets.level.player_spawn.yaw,
		objects        = make([dynamic]Object, 0, MAX_OBJECTS, allocator),
		suzanne_mesh   = gpu.suzanne_handle,
		map_mesh       = gpu.map_handle,
		next_object_id = 1,
	}
	scene_add_player(&scene, LOCAL_PLAYER_ID, scene.player_spawn_position, scene.player_spawn_yaw)
	scene_physics_assets_create(&scene, &assets.collision_mesh, &assets.level.render_mesh)
	scene_create_map(&scene)
	scene_spawn_level_entities(&scene, &assets.level.source)
	return scene
}

scene_add_player :: proc(scene: ^Scene, player_id: u32, position: Vec3, yaw: f32) -> ^PlayerController {
	assert(len(scene.players) < cap(scene.players))
	append(&scene.players, ScenePlayer{id = player_id, controller = player_create(position, yaw)})
	return &scene.players[len(scene.players) - 1].controller
}

scene_ensure_player :: proc(scene: ^Scene, player_id: u32, position := PLAYER_SPEC.spawn_position, yaw: f32 = 0) -> ^PlayerController {
	if player := scene_player(scene, player_id); player != nil {
		return player
	}
	return scene_add_player(scene, player_id, position, yaw)
}

scene_reset_player :: proc(scene: ^Scene, player_id: u32, position := PLAYER_SPEC.spawn_position, yaw: f32 = 0) -> ^PlayerController {
	player := scene_ensure_player(scene, player_id, position, yaw)
	player^ = player_create(position, yaw)
	return player
}

scene_reset_player_to_spawn :: proc(scene: ^Scene, player_id: u32) -> ^PlayerController {
	return scene_reset_player(scene, player_id, scene.player_spawn_position, scene.player_spawn_yaw)
}

scene_player :: proc(scene: ^Scene, player_id: u32) -> ^PlayerController {
	for &player in scene.players {
		if player.id == player_id {
			return &player.controller
		}
	}
	return nil
}

scene_camera_player :: proc(scene: ^Scene) -> ^PlayerController {
	player := scene_player(scene, scene.camera_player_id)
	if player != nil {
		return player
	}
	if len(scene.players) > 0 {
		return &scene.players[0].controller
	}
	return nil
}

scene_destroy :: proc(scene: ^Scene) {
	delete(scene.players)
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
		case .Map, .RemotePlayer:
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
	scene.player_spawn_position = level.player_spawn.position
	scene.player_spawn_yaw = level.player_spawn.yaw
	for &player in scene.players {
		player.controller = player_create(scene.player_spawn_position, scene.player_spawn_yaw)
	}
	scene_clear_level_entities(scene)
	scene_spawn_level_entities(scene, &level.source)
	scene.accumulator = 0
}

scene_update :: proc(
	scene: ^Scene,
	player_id: u32,
	move_input: PlayerMoveInput,
	look_input: PlayerLookInput,
	delta_time: f32,
) {
	update_start := scene_profile_counter_now()
	scene.profile = {}

	player := scene_player(scene, player_id)
	if player == nil {
		return
	}
	player_apply_look(player, look_input)

	scene.accumulator += min(delta_time, 0.25)
	for scene.accumulator >= PHYSICS_STEP_TIME {
		fixed_start := scene_profile_counter_now()
		scene_fixed_update(scene, player_id, move_input, PHYSICS_STEP_TIME)
		scene.profile.fixed_update_ms += scene_profile_elapsed_ms(fixed_start)
		scene.profile.fixed_steps += 1
		scene.accumulator -= PHYSICS_STEP_TIME
	}

	scene.profile.update_ms = scene_profile_elapsed_ms(update_start)
	scene_profile_log_if_needed(scene, delta_time)
}

scene_update_from_user_cmd :: proc(scene: ^Scene, player_id: u32, cmd: protocol.User_Cmd, delta_time: f32) {
	move := PlayerMoveInput{
		move_forward = cmd.move_forward,
		move_right = cmd.move_right,
		jump_held = (cmd.buttons & protocol.BUTTON_JUMP) != 0,
	}
	player := scene_player(scene, player_id)
	if player == nil {
		return
	}
	player.yaw = cmd.yaw
	player.pitch = cmd.pitch
	scene_update(scene, player_id, move, {}, delta_time)
}

scene_fixed_update :: proc(scene: ^Scene, player_id: u32, input: PlayerMoveInput, step_time: f32) {
	think_start := scene_profile_counter_now()
	scene_run_think(scene, step_time)
	scene.profile.think_ms += scene_profile_elapsed_ms(think_start)

	player_start := scene_profile_counter_now()
	player := scene_player(scene, player_id)
	if player == nil {
		return
	}
	move := player_update(player, &scene.physics, input, step_time)
	scene.profile.player_ms += scene_profile_elapsed_ms(player_start)

	touch_start := scene_profile_counter_now()
	scene_touch_player(scene, player, move)
	scene.profile.touch_ms += scene_profile_elapsed_ms(touch_start)

	physics_start := scene_profile_counter_now()
	engine.physics_step(&scene.physics)
	scene.profile.box3d_step_ms += scene_profile_elapsed_ms(physics_start)
	scene.profile.box3d = b3.World_GetProfile(scene.physics.id)
	scene.profile.counters = b3.World_GetCounters(scene.physics.id)
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

scene_upsert_remote_player :: proc(scene: ^Scene, player_id: u32, position: Vec3, yaw: f32) {
	for &object in scene.objects {
		if object.kind == .RemotePlayer && object.net_id == player_id {
			object.transform.position = position
			object.transform.yaw = yaw
			object.render.visible = true
			return
		}
	}

	if len(scene.objects) >= cap(scene.objects) {
		log.warnf("Cannot add remote player %d: object capacity reached", player_id)
		return
	}

	_ = scene_add_object(
		scene,
		Object {
			name = "RemotePlayer",
			kind = .RemotePlayer,
			net_id = player_id,
			transform = {position = position, yaw = yaw},
			render = {mesh = scene.suzanne_mesh, visible = true},
			physics = {enabled = false},
		},
	)
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

scene_debug_hud_data :: proc(scene: ^Scene) -> DebugHudData {
	player := scene_camera_player(scene)
	player_position: Vec3
	player_velocity: Vec3
	player_grounded := false
	if player != nil {
		player_position = player.position
		player_velocity = player.velocity
		player_grounded = player.grounded
	}
	return {
		enabled = true,
		object_count = len(scene.objects),
		suzanne_count = scene_count_objects(scene, .Suzanne),
		object_capacity = cap(scene.objects),
		player_position = player_position,
		player_velocity = player_velocity,
		player_grounded = player_grounded,
		fixed_steps = scene.profile.fixed_steps,
		physics_step_ms = scene.profile.box3d.step,
		physics_collide_ms = scene.profile.box3d.collide,
		physics_solve_ms = scene.profile.box3d.solve,
		physics_pairs_ms = scene.profile.box3d.pairs,
		physics_contacts = int(scene.profile.counters.contactCount),
		physics_awake_contacts = int(scene.profile.counters.awakeContactCount),
		physics_tree_height = int(scene.profile.counters.treeHeight),
	}
}

scene_profile_log_if_needed :: proc(scene: ^Scene, delta_time: f32) {
	scene.profile_log_timer += delta_time
	if scene.profile_log_timer < 1.0 || len(scene.objects) < 200 {
		return
	}
	scene.profile_log_timer = 0
	p := scene.profile
	log.debugf(
		"profile objects=%d suzannes=%d update=%.2fms fixed=%.2fms steps=%d think=%.2fms player=%.2fms touch=%.2fms box3d_wall=%.2fms box3d_step=%.2fms pairs=%.2fms collide=%.2fms solve=%.2fms contacts=%d awake_contacts=%d tree=%d sat=%d",
		len(scene.objects),
		scene_count_objects(scene, .Suzanne),
		p.update_ms,
		p.fixed_update_ms,
		p.fixed_steps,
		p.think_ms,
		p.player_ms,
		p.touch_ms,
		p.box3d_step_ms,
		p.box3d.step,
		p.box3d.pairs,
		p.box3d.collide,
		p.box3d.solve,
		p.counters.contactCount,
		p.counters.awakeContactCount,
		p.counters.treeHeight,
		p.counters.satCallCount,
	)
}

scene_profile_counter_now :: proc() -> u64 {
	return sdl.GetPerformanceCounter()
}

scene_profile_elapsed_ms :: proc(start_counter: u64) -> f32 {
	end_counter := sdl.GetPerformanceCounter()
	frequency := sdl.GetPerformanceFrequency()
	return f32(end_counter - start_counter) * 1000.0 / f32(frequency)
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
	player := scene_camera_player(scene)
	view: matrix[4, 4]f32
	if player != nil {
		view = player_view_matrix(player)
	}
	return RenderPassGlobals {
		view        = view,
		proj        = engine.matrix4_perspective_z0_f32(linalg.to_radians(f32(70)), aspect, 0.1, 100),
		environment = scene_render_environment(),
	}
}

scene_render_environment :: proc() -> RenderEnvironment {
	return {
		fog_color         = {0.776471, 0.866667, 0.909804, 1.0}, // Flexoki blue-100
		sky_top_color     = {0.262745, 0.521569, 0.745098, 1.0}, // Flexoki blue-400
		sky_horizon_color = {0.776471, 0.866667, 0.909804, 1.0}, // Flexoki blue-100
		fog_distances     = {55.0, 95.0, 0.0, 0.0},
	}
}

scene_object_model_matrix :: proc(object: ^Object) -> matrix[4, 4]f32 {
	if object.physics.sync_transform {
		return engine.physics_body_matrix(object.physics.body)
	}
	return scene_transform_matrix(object.transform)
}

scene_transform_matrix :: proc(transform: Transform) -> matrix[4, 4]f32 {
	rotation := linalg.quaternion_from_euler_angle_y_f32(transform.yaw)
	return linalg.matrix4_from_trs_f32(transform.position, rotation, {1, 1, 1})
}

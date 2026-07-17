package game

import engine "../engine"
import protocol "../protocol"
import "base:runtime"
import "core:log"
import "core:math/linalg"
import "core:strings"
import b3 "vendor:box3d"

SPAWN_INTERVAL :: f32(0.12)
MAX_OBJECTS :: 2048
MAX_PLAYERS :: 32
LOCAL_PLAYER_ID :: u32(1)
REMOTE_PLAYER_SAMPLE_CAPACITY :: 8
REMOTE_INTERPOLATION_DELAY_MS :: 100
REMOTE_INTERPOLATION_DELAY_SECONDS :: f32(REMOTE_INTERPOLATION_DELAY_MS) / 1000.0
REMOTE_INTERPOLATION_DELAY_TICKS :: u32((REMOTE_INTERPOLATION_DELAY_MS * NET_SERVER_TICK_HZ + 999) / 1000)

PlayerTickInput :: struct {
	player_id: u32,
	move:      PlayerMoveInput,
	yaw:       f32,
	pitch:     f32,
}

Scene :: struct {
	allocator:      runtime.Allocator,
	physics:        PhysicsWorld,
	physics_assets: ScenePhysicsAssets,
	players:        [dynamic]ScenePlayer,
	camera_player_id: u32,
	player_spawn_position: Vec3,
	player_spawn_yaw: f32,
	objects:        [dynamic]Object,
	prop_meshes:    [MAX_PROP_ASSETS]MeshHandle,
	prop_model_paths: [MAX_PROP_ASSETS]string,
	prop_asset_count: int,
	map_mesh:       MeshHandle,
	next_object_id: ObjectId,
	accumulator:    f32,
	remote_render_tick: f32,
	profile:        SceneProfile,
	profile_log_timer: f32,
}

ScenePlayer :: struct {
	id:         u32,
	controller: PlayerController,
	remote_sample_count: int,
	remote_samples: [REMOTE_PLAYER_SAMPLE_CAPACITY]RemotePlayerSample,
}

RemotePlayerSample :: struct {
	server_tick: u32,
	position:    Vec3,
	yaw:         f32,
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
	Prop,
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
	shape:            b3.ShapeId,
	enabled:          bool,
	sync_transform:   bool,
	linear_velocity:  Vec3,
	angular_velocity: Vec3,
}

ThinkKind :: enum {
	None,
	SpawnerProp,
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
	target_position: Vec3,
	target_yaw:      f32,
}

Object :: struct {
	id:        ObjectId,
	name:      string,
	kind:      ObjectKind,
	transform: Transform,
	render_rotation: linalg.Quaternionf32,
	render:    RenderObject,
	physics:   PhysicsObject,
	think:     ThinkObject,
	touch:     TouchObject,
	replica:   ReplicatedObject,
	prop_asset_index: u16,
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
		prop_meshes    = gpu.prop_handles,
		prop_asset_count = len(assets.prop_assets),
		map_mesh       = gpu.map_handle,
		next_object_id = 1,
	}
	for &asset, i in assets.prop_assets {
		scene.prop_model_paths[i] = strings.clone(asset.model_path, allocator)
	}
	scene_add_player(&scene, LOCAL_PLAYER_ID, scene.player_spawn_position, scene.player_spawn_yaw)
	scene_physics_assets_create(&scene, assets)
	scene_create_map(&scene)
	scene_spawn_level_entities(&scene, &assets.level.source)
	return scene
}

scene_add_player :: proc(scene: ^Scene, player_id: u32, position: Vec3, yaw: f32) -> ^PlayerController {
	assert(len(scene.players) < cap(scene.players))
	append(&scene.players, ScenePlayer{id = player_id, controller = player_create(position, yaw)})
	return &scene.players[len(scene.players) - 1].controller
}

scene_add_player_record :: proc(scene: ^Scene, player_id: u32, position: Vec3, yaw: f32) -> ^ScenePlayer {
	assert(len(scene.players) < cap(scene.players))
	append(&scene.players, ScenePlayer{id = player_id, controller = player_create(position, yaw)})
	return &scene.players[len(scene.players) - 1]
}

scene_ensure_player :: proc(scene: ^Scene, player_id: u32, position := PLAYER_SPEC.spawn_position, yaw: f32 = 0) -> ^PlayerController {
	if player := scene_player(scene, player_id); player != nil {
		return player
	}
	return scene_add_player(scene, player_id, position, yaw)
}

scene_ensure_player_record :: proc(scene: ^Scene, player_id: u32, position := PLAYER_SPEC.spawn_position, yaw: f32 = 0) -> ^ScenePlayer {
	if player := scene_player_record(scene, player_id); player != nil {
		return player
	}
	return scene_add_player_record(scene, player_id, position, yaw)
}

scene_reset_player :: proc(scene: ^Scene, player_id: u32, position := PLAYER_SPEC.spawn_position, yaw: f32 = 0) -> ^PlayerController {
	player := scene_ensure_player_record(scene, player_id, position, yaw)
	player.controller = player_create(position, yaw)
	player.remote_sample_count = 0
	return &player.controller
}

scene_reset_player_to_spawn :: proc(scene: ^Scene, player_id: u32) -> ^PlayerController {
	return scene_reset_player(scene, player_id, scene.player_spawn_position, scene.player_spawn_yaw)
}

scene_player :: proc(scene: ^Scene, player_id: u32) -> ^PlayerController {
	if player := scene_player_record(scene, player_id); player != nil {
		return &player.controller
	}
	return nil
}

scene_player_record :: proc(scene: ^Scene, player_id: u32) -> ^ScenePlayer {
	for &player in scene.players {
		if player.id == player_id {
			return &player
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
	for i in 0..<scene.prop_asset_count {
		delete(scene.prop_model_paths[i], scene.allocator)
	}
	delete(scene.players)
	delete(scene.objects)
	scene_physics_assets_destroy(scene)
	engine.physics_destroy(&scene.physics)
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
	scene_physics_assets_destroy(scene)
	engine.physics_destroy(&scene.physics)
}

scene_rebuild_after_hot_reload :: proc(scene: ^Scene, fs: ^GameFS, map_qpath: string) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	assets := scene_assets_load(fs, map_qpath)
	defer scene_assets_destroy(&assets)

	scene.physics = engine.physics_create()
	scene_physics_assets_create(scene, &assets)
	for &object, index in scene.objects {
		switch object.kind {
		case .Prop:
			body := scene_physics_create_prop_body(scene, object.transform.position, object.prop_asset_index, int(index))
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

scene_reload_assets :: proc(scene: ^Scene, assets: ^LoadedSceneAssets, gpu: SceneGpuResources) {
	scene_clear_level_entities(scene)
	scene_physics_assets_destroy(scene)
	for i in 0..<scene.prop_asset_count {
		delete(scene.prop_model_paths[i], scene.allocator)
		scene.prop_model_paths[i] = ""
	}

	scene.prop_meshes = gpu.prop_handles
	scene.prop_asset_count = len(assets.prop_assets)
	scene.map_mesh = gpu.map_handle
	for &asset, i in assets.prop_assets {
		scene.prop_model_paths[i] = strings.clone(asset.model_path, scene.allocator)
	}
	scene_physics_assets_create(scene, assets)

	scene.player_spawn_position = assets.level.player_spawn.position
	scene.player_spawn_yaw = assets.level.player_spawn.yaw
	for &player in scene.players {
		player.controller = player_create(scene.player_spawn_position, scene.player_spawn_yaw)
		player.remote_sample_count = 0
	}
	scene_create_map(scene)
	scene_spawn_level_entities(scene, &assets.level.source)
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

player_move_input_from_user_cmd :: proc(cmd: protocol.User_Cmd) -> PlayerMoveInput {
	return {
		move_forward = cmd.move_forward,
		move_right = cmd.move_right,
		jump_held = (cmd.buttons & protocol.BUTTON_JUMP) != 0,
	}
}

scene_fixed_update_players :: proc(scene: ^Scene, inputs: []PlayerTickInput, step_time: f32) {
	think_start := scene_profile_counter_now()
	scene_run_think(scene, step_time)
	scene.profile.think_ms += scene_profile_elapsed_ms(think_start)

	player_start := scene_profile_counter_now()
	for input in inputs {
		player := scene_player(scene, input.player_id)
		if player == nil {
			continue
		}
		player.yaw = input.yaw
		player.pitch = input.pitch
		move := player_update(player, &scene.physics, input.move, step_time)
		touch_start := scene_profile_counter_now()
		scene_touch_player(scene, player, move)
		scene.profile.touch_ms += scene_profile_elapsed_ms(touch_start)
	}
	scene.profile.player_ms += scene_profile_elapsed_ms(player_start)

	scene_step_physics(scene, step_time)
	scene.profile.fixed_steps += 1
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

	scene_step_physics(scene, step_time)
}

scene_step_physics :: proc(scene: ^Scene, elapsed_time: f32) {
	physics_start := scene_profile_counter_now()
	remaining := elapsed_time
	for remaining > 0 {
		step_time := min(remaining, PHYSICS_STEP_TIME)
		engine.physics_step(&scene.physics, step_time)
		remaining -= step_time
	}
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
			render_rotation = linalg.QUATERNIONF32_IDENTITY,
			render = {mesh = scene.map_mesh, visible = true},
			physics = {enabled = false},
		},
	)
}

scene_spawn_prop :: proc(scene: ^Scene, position: Vec3, prop_asset_index: u16 = 0, authority := ReplicatedPropAuthority.ServerAuthoritative) -> bool {
	i := scene_count_objects(scene, .Prop)
	if len(scene.objects) >= cap(scene.objects) {
		return false
	}
	mesh := scene_prop_mesh(scene, prop_asset_index)

	body := scene_physics_create_prop_body(scene, position, prop_asset_index, i)
	scene_add_object(
		scene,
		Object {
			name = "Prop",
			kind = .Prop,
			transform = {position = position},
			render_rotation = linalg.QUATERNIONF32_IDENTITY,
			render = {mesh = mesh, visible = true},
			physics = {
				body = body,
				enabled = true,
				sync_transform = true,
				angular_velocity = scene_physics_prop_angular_velocity(i),
			},
			replica = {kind = .Prop, authority = authority},
			prop_asset_index = prop_asset_index,
		},
	)
	return true
}

scene_prop_mesh :: proc(scene: ^Scene, prop_asset_index: u16) -> MeshHandle {
	index := int(prop_asset_index)
	if index < 0 || index >= scene.prop_asset_count {
		index = 0
	}
	return scene.prop_meshes[index]
}

scene_prop_asset_index_by_model :: proc(scene: ^Scene, model_path: string) -> u16 {
	for i in 0..<scene.prop_asset_count {
		if scene.prop_model_paths[i] == model_path {
			return u16(i)
		}
	}
	log.warnf("Prop model not loaded for scene, falling back to default: %s", model_path)
	return 0
}

scene_upsert_remote_player :: proc(scene: ^Scene, player_id: u32, position: Vec3, yaw: f32) {
	player := scene_ensure_player(scene, player_id, position, yaw)
	player.position = position
	player.yaw = yaw
}

scene_upsert_remote_player_sample :: proc(scene: ^Scene, player_id: u32, position: Vec3, yaw: f32, server_tick: u32) {
	player := scene_ensure_player_record(scene, player_id, position, yaw)
	player.controller.position = position
	player.controller.yaw = yaw
	scene_player_add_remote_sample(player, {server_tick = server_tick, position = position, yaw = yaw})
}

scene_player_add_remote_sample :: proc(player: ^ScenePlayer, sample: RemotePlayerSample) {
	for i in 0..<player.remote_sample_count {
		if player.remote_samples[i].server_tick == sample.server_tick {
			player.remote_samples[i] = sample
			return
		}
	}

	if player.remote_sample_count >= REMOTE_PLAYER_SAMPLE_CAPACITY {
		copy(player.remote_samples[:], player.remote_samples[1:])
		player.remote_sample_count -= 1
	}

	insert_at := player.remote_sample_count
	for insert_at > 0 && player.remote_samples[insert_at - 1].server_tick > sample.server_tick {
		player.remote_samples[insert_at] = player.remote_samples[insert_at - 1]
		insert_at -= 1
	}
	player.remote_samples[insert_at] = sample
	player.remote_sample_count += 1
}

scene_player_render_transform :: proc(scene: ^Scene, player: ^ScenePlayer) -> Transform {
	if player.id == scene.camera_player_id || player.remote_sample_count == 0 {
		return {position = player.controller.position, yaw = player.controller.yaw}
	}
	return scene_player_interpolated_transform(player, scene.remote_render_tick)
}

scene_player_interpolated_transform :: proc(player: ^ScenePlayer, render_tick: f32) -> Transform {
	if player.remote_sample_count == 0 {
		return {position = player.controller.position, yaw = player.controller.yaw}
	}
	if player.remote_sample_count == 1 || render_tick <= f32(player.remote_samples[0].server_tick) {
		sample := player.remote_samples[0]
		return {position = sample.position, yaw = sample.yaw}
	}

	last_index := player.remote_sample_count - 1
	if render_tick >= f32(player.remote_samples[last_index].server_tick) {
		sample := player.remote_samples[last_index]
		return {position = sample.position, yaw = sample.yaw}
	}

	for i in 0..<last_index {
		from := player.remote_samples[i]
		to := player.remote_samples[i + 1]
		if render_tick >= f32(from.server_tick) && render_tick <= f32(to.server_tick) {
			span := to.server_tick - from.server_tick
			if span == 0 {
				return {position = to.position, yaw = to.yaw}
			}
			t := (render_tick - f32(from.server_tick)) / f32(span)
			return {
				position = scene_lerp_vec3(from.position, to.position, t),
				yaw = scene_lerp_f32(from.yaw, to.yaw, t),
			}
		}
	}

	sample := player.remote_samples[last_index]
	return {position = sample.position, yaw = sample.yaw}
}

scene_lerp_vec3 :: proc(a, b: Vec3, t: f32) -> Vec3 {
	return a + (b - a) * t
}

scene_lerp_f32 :: proc(a, b: f32, t: f32) -> f32 {
	return a + (b - a) * t
}

scene_add_object :: proc(scene: ^Scene, object: Object) -> ObjectId {
	assert(len(scene.objects) < cap(scene.objects))
	id := scene.next_object_id
	scene.next_object_id = ObjectId(u32(scene.next_object_id) + 1)
	stored := object
	stored.id = id
	if stored.replica.net_id == 0 && stored.replica.kind != .None && stored.replica.authority == .ServerAuthoritative {
		stored.replica.net_id = protocol.NetId(u32(id))
	}
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

scene_object :: proc(scene: ^Scene, object_id: ObjectId) -> ^Object {
	for &object in scene.objects {
		if object.id == object_id {
			return &object
		}
	}
	return nil
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
		prop_count = scene_count_objects(scene, .Prop),
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
		"profile objects=%d props=%d update=%.2fms fixed=%.2fms steps=%d think=%.2fms player=%.2fms touch=%.2fms box3d_wall=%.2fms box3d_step=%.2fms pairs=%.2fms collide=%.2fms solve=%.2fms contacts=%d awake_contacts=%d tree=%d sat=%d",
		len(scene.objects),
		scene_count_objects(scene, .Prop),
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
	return engine.performance_counter_now()
}

scene_profile_elapsed_ms :: proc(start_counter: u64) -> f32 {
	return engine.performance_elapsed_ms(start_counter)
}

scene_collect_render_items :: proc(scene: ^Scene, items: ^[dynamic]RenderItem) -> []RenderItem {
	clear(items)
	for &object in scene.objects {
		if !object.render.visible {
			continue
		}
		model := scene_object_model_matrix(scene, &object)
		append(items, RenderItem{mesh = object.render.mesh, model = model})
	}
	for &player in scene.players {
		if player.id == scene.camera_player_id {
			continue
		}
		model := scene_transform_matrix(scene_player_render_transform(scene, &player))
		append(items, RenderItem{mesh = scene_prop_mesh(scene, 0), model = model})
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

scene_object_model_matrix :: proc(scene: ^Scene, object: ^Object) -> matrix[4, 4]f32 {
	if object.physics.sync_transform {
		return engine.physics_body_matrix(object.physics.body)
	}
	if object.kind == .Prop {
		position, rotation := replicated_transform_at_tick(&object.replica.transform_buffer, object.transform.position, object.render_rotation, scene.remote_render_tick)
		return linalg.matrix4_from_trs_f32(position, rotation, {1, 1, 1})
	}
	return scene_transform_matrix(object.transform)
}

scene_transform_matrix :: proc(transform: Transform) -> matrix[4, 4]f32 {
	rotation := linalg.quaternion_from_euler_angle_y_f32(transform.yaw)
	return linalg.matrix4_from_trs_f32(transform.position, rotation, {1, 1, 1})
}

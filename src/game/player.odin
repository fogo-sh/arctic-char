package game

import "core:math"
import "core:math/linalg"
import engine "../engine"
import b3 "vendor:box3d"
import sdl "vendor:sdl3"

PlayerMoveConfig :: struct {
	gravity:            f32,
	max_speed:          f32,
	stop_speed:         f32,
	ground_accel:       f32,
	air_accel:          f32,
	friction:           f32,
	jump_velocity:      f32,
	air_wishspeed_cap:  f32,
}

PlayerSpec :: struct {
	move:                 PlayerMoveConfig,
	spawn_position:       Vec3,
	eye_height:           f32,
	mouse_sensitivity:    f32,
	hull_mins:            Vec3,
	hull_maxs:            Vec3,
	capsule_radius:       f32,
	capsule_half_height:  f32,
	capsule_center_y:     f32,
	step_height:          f32,
	walkable_min_y:       f32,
}

PlayerController :: struct {
	position:       Vec3,
	spawn_position: Vec3,
	velocity:       Vec3,
	yaw:            f32,
	spawn_yaw:      f32,
	pitch:          f32,
	grounded:       bool,
	ground_normal:  Vec3,
}

PlayerMoveInput :: struct {
	move_forward: f32,
	move_right:   f32,
	jump_held:    bool,
}

PlayerLookInput :: struct {
	look_delta: [2]f32,
}

PlayerMoveResult :: struct {
	old_position: Vec3,
	new_position: Vec3,
	velocity:     Vec3,
	grounded:     bool,
}

PLAYER_SPEC :: PlayerSpec{
	move = {
		gravity = 800.0 * QU_TO_M,
		max_speed = 320.0 * QU_TO_M,
		stop_speed = 100.0 * QU_TO_M,
		ground_accel = 10.0,
		air_accel = 10.0,
		friction = 6.0,
		jump_velocity = 270.0 * QU_TO_M,
		air_wishspeed_cap = 30.0 * QU_TO_M,
	},
	spawn_position = {0, 0.9, 8.0},
	eye_height = 22.0 * QU_TO_M,
	mouse_sensitivity = 0.0022,
	hull_mins = {-16.0 * QU_TO_M, -24.0 * QU_TO_M, -16.0 * QU_TO_M},
	hull_maxs = {16.0 * QU_TO_M, 32.0 * QU_TO_M, 16.0 * QU_TO_M},
	capsule_radius = 16.0 * QU_TO_M,
	capsule_half_height = (56.0 * QU_TO_M) * 0.5 - 16.0 * QU_TO_M,
	capsule_center_y = 4.0 * QU_TO_M,
	step_height = 18.0 * QU_TO_M,
	walkable_min_y = 0.7,
}

PLAYER_DEFAULT_SPAWN :: MapPlayerSpawn{position = PLAYER_SPEC.spawn_position, yaw = 0}

player_create :: proc(position := PLAYER_SPEC.spawn_position, yaw: f32 = 0) -> PlayerController {
	return PlayerController{
		position = position,
		spawn_position = position,
		yaw = yaw,
		spawn_yaw = yaw,
		ground_normal = {0, 1, 0},
	}
}

player_input_from_engine :: proc(input: InputState) -> (move: PlayerMoveInput, look: PlayerLookInput) {
	if engine.input_key_down(input, sdl.Scancode.W) do move.move_forward += 1
	if engine.input_key_down(input, sdl.Scancode.S) do move.move_forward -= 1
	if engine.input_key_down(input, sdl.Scancode.D) do move.move_right += 1
	if engine.input_key_down(input, sdl.Scancode.A) do move.move_right -= 1
	move.jump_held = engine.input_key_down(input, sdl.Scancode.SPACE)
	if input.mouse_captured {
		look.look_delta = input.mouse_delta
	}
	return
}

player_apply_look :: proc(player: ^PlayerController, input: PlayerLookInput) {
	player.yaw, player.pitch = player_look_angles_after_input(player^, input)
}

player_look_angles_after_input :: proc(player: PlayerController, input: PlayerLookInput) -> (yaw, pitch: f32) {
	spec := PLAYER_SPEC
	yaw = player.yaw + input.look_delta.x * spec.mouse_sensitivity
	pitch = player.pitch - input.look_delta.y * spec.mouse_sensitivity
	pitch = math.clamp(pitch, linalg.to_radians(f32(-89)), linalg.to_radians(f32(89)))
	return
}

player_update :: proc(player: ^PlayerController, physics: ^PhysicsWorld, input: PlayerMoveInput, delta_time: f32) -> PlayerMoveResult {
	old_position := player.position
	spec := PLAYER_SPEC
	forward, right := player_flat_basis(player.yaw)
	wish_dir := forward * input.move_forward + right * input.move_right
	wish_dir = linalg.normalize0(wish_dir)

	config := spec.move
	if player.grounded {
		if input.jump_held {
			player.velocity.y = config.jump_velocity
			player.grounded = false
			player_air_accelerate(player, wish_dir, config.max_speed, config.air_accel, delta_time, config.air_wishspeed_cap)
		} else {
			player_apply_friction(player, physics, spec, config, delta_time)
			player_accelerate(player, wish_dir, config.max_speed, config.ground_accel, delta_time)
			player.velocity.y = min(player.velocity.y, f32(0))
		}
	} else {
		player.velocity.y -= config.gravity * delta_time
		player_air_accelerate(player, wish_dir, config.max_speed, config.air_accel, delta_time, config.air_wishspeed_cap)
	}

	displacement := player.velocity * delta_time
	player.position, player.velocity, player.grounded, player.ground_normal = player_mover_move(
		physics,
		spec,
		player.position,
		displacement,
		player.velocity,
		player.grounded,
	)

	return {
		old_position = old_position,
		new_position = player.position,
		velocity = player.velocity,
		grounded = player.grounded,
	}
}

player_teleport :: proc(player: ^PlayerController, position: Vec3, yaw: f32) {
	player.position = position
	player.velocity = {}
	player.yaw = yaw
	player.pitch = 0
	player.grounded = false
	player.ground_normal = {0, 1, 0}
}

player_apply_friction :: proc(player: ^PlayerController, physics: ^PhysicsWorld, spec: PlayerSpec, config: PlayerMoveConfig, delta_time: f32) {
	horizontal := Vec3{player.velocity.x, 0, player.velocity.z}
	speed := linalg.length(horizontal)
	if speed < QU_TO_M {
		player.velocity.x = 0
		player.velocity.z = 0
		return
	}

	friction := config.friction
	if player_ledge_friction_applies(player, physics, spec, horizontal / speed) {
		friction *= 2
	}

	control := max(config.stop_speed, speed)
	drop := control * friction * delta_time
	new_speed := max(speed - drop, f32(0)) / speed
	player.velocity.x *= new_speed
	player.velocity.z *= new_speed
}

player_ledge_friction_applies :: proc(player: ^PlayerController, physics: ^PhysicsWorld, spec: PlayerSpec, direction: Vec3) -> bool {
	start := player.position + direction * (16.0 * QU_TO_M)
	start.y += spec.hull_mins.y
	result := b3.World_CastRayClosest(
		physics.id,
		b3.Pos(start),
		{0, -(34.0 * QU_TO_M), 0},
		physics_player_query_filter(),
	)
	return !result.hit
}

player_accelerate :: proc(player: ^PlayerController, dir: Vec3, wish_speed, accel, delta_time: f32) {
	if linalg.length(dir) == 0 {
		return
	}
	current_speed := linalg.dot(player.velocity, dir)
	add_speed := wish_speed - current_speed
	if add_speed <= 0 {
		return
	}
	accel_speed := accel * wish_speed * delta_time
	if accel_speed > add_speed {
		accel_speed = add_speed
	}
	player.velocity += dir * accel_speed
}

player_air_accelerate :: proc(player: ^PlayerController, dir: Vec3, wish_speed, accel, delta_time, wish_speed_cap: f32) {
	wish_spd := min(wish_speed, wish_speed_cap)
	if linalg.length(dir) == 0 {
		return
	}
	current_speed := linalg.dot(player.velocity, dir)
	add_speed := wish_spd - current_speed
	if add_speed <= 0 {
		return
	}
	accel_speed := accel * wish_speed * delta_time
	if accel_speed > add_speed {
		accel_speed = add_speed
	}
	player.velocity += dir * accel_speed
}

player_flat_basis :: proc(yaw: f32) -> (forward, right: Vec3) {
	forward = {math.sin(yaw), 0, -math.cos(yaw)}
	right = {math.cos(yaw), 0, math.sin(yaw)}
	return
}

player_look_forward :: proc(player: ^PlayerController) -> Vec3 {
	cp := math.cos(player.pitch)
	return Vec3{
		math.sin(player.yaw) * cp,
		math.sin(player.pitch),
		-math.cos(player.yaw) * cp,
	}
}

player_eye_position :: proc(player: ^PlayerController) -> Vec3 {
	return player.position + Vec3{0, PLAYER_SPEC.eye_height, 0}
}

player_view_matrix :: proc(player: ^PlayerController) -> matrix[4, 4]f32 {
	eye := player_eye_position(player)
	return linalg.matrix4_look_at_f32(eye, eye + player_look_forward(player), {0, 1, 0})
}

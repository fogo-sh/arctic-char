package main

import "core:math"
import "core:math/linalg"
import sdl "vendor:sdl3"

CameraMode :: enum {
	Noclip,
	Player,
}

CameraPerspective :: enum {
	Perspective,
	Orthographic,
}

Camera :: struct {
	mode:              CameraMode,
	position:          [3]f32,
	is_locked:         bool,
	perspective:       CameraPerspective,
	yaw:               f32,
	pitch:             f32,
	mouse_sensitivity: f32,
}

camera_update :: proc(camera: ^Camera, delta_time: f32, movement: Movement) -> matrix[4, 4]f32 {
	view_mat := linalg.MATRIX4F32_IDENTITY

	if camera.mode == .Noclip {
		sin_y := math.sin(camera.yaw)
		cos_y := math.cos(camera.yaw)
		sin_p := math.sin(camera.pitch)
		cos_p := math.cos(camera.pitch)

		forward_vec := [3]f32{-sin_y * cos_p, sin_p, -cos_y * cos_p}
		right_vec := [3]f32{cos_y, 0.0, -sin_y}

		move_speed := f32(5.0)
		if movement.shift {
			move_speed = 20.0
		}
		dt_move := move_speed * delta_time

		if movement.forward {
			camera.position[0] += forward_vec[0] * dt_move
			camera.position[1] += forward_vec[1] * dt_move
			camera.position[2] += forward_vec[2] * dt_move
		}
		if movement.backward {
			camera.position[0] -= forward_vec[0] * dt_move
			camera.position[1] -= forward_vec[1] * dt_move
			camera.position[2] -= forward_vec[2] * dt_move
		}
		if movement.left {
			camera.position[0] -= right_vec[0] * dt_move
			camera.position[1] -= right_vec[1] * dt_move
			camera.position[2] -= right_vec[2] * dt_move
		}
		if movement.right {
			camera.position[0] += right_vec[0] * dt_move
			camera.position[1] += right_vec[1] * dt_move
			camera.position[2] += right_vec[2] * dt_move
		}
		if movement.up {
			camera.position[1] += dt_move
		}
		if movement.down {
			camera.position[1] -= dt_move
		}

		view_mat = linalg.matrix4_rotate_f32(-camera.yaw, {0, 1, 0}) * view_mat
		view_mat = linalg.matrix4_rotate_f32(-camera.pitch, {1, 0, 0}) * view_mat

		view_mat =
			view_mat *
			linalg.matrix4_translate_f32(
				{-camera.position[0], -camera.position[1], -camera.position[2]},
			)
	}

	return view_mat
}

camera_handle_mouse_motion :: proc(camera: ^Camera, x_rel: f32, y_rel: f32) {
	if camera.mode == .Noclip && camera.is_locked {
		camera.yaw -= x_rel * camera.mouse_sensitivity
		camera.pitch -= y_rel * camera.mouse_sensitivity

		if camera.pitch > linalg.to_radians(f32(89)) {
			camera.pitch = linalg.to_radians(f32(89))
		} else if camera.pitch < linalg.to_radians(f32(-89)) {
			camera.pitch = linalg.to_radians(f32(-89))
		}
	}
}

camera_handle_mouse_button :: proc(
	camera: ^Camera,
	button: i32,
	pressed: bool,
	window: ^sdl.Window,
) -> bool {
	if button == 1 && pressed {
		return camera_lock(camera, window)
	}
	return false
}

camera_lock :: proc(camera: ^Camera, window: ^sdl.Window) -> bool {
	ok := sdl.SetWindowRelativeMouseMode(window, true)
	camera.is_locked = ok
	return ok
}

camera_unlock :: proc(camera: ^Camera, window: ^sdl.Window) -> bool {
	ok := sdl.SetWindowRelativeMouseMode(window, false)
	camera.is_locked = false
	return ok
}

camera_get_projection :: proc(
	camera: ^Camera,
	window_width: i32,
	window_height: i32,
) -> matrix[4, 4]f32 {
	aspect_ratio := f32(window_width) / f32(window_height)

	if camera.perspective == .Orthographic {
		ortho_width := f32(20.0)
		ortho_height := ortho_width / aspect_ratio
		return matrix4_orthographic_f32(
			-ortho_width / 2,
			ortho_width / 2,
			-ortho_height / 2,
			ortho_height / 2,
			0.1,
			1000.0,
		)
	} else {
		return linalg.matrix4_perspective_f32(linalg.to_radians(f32(70)), aspect_ratio, 0.1, 1000)
	}
}

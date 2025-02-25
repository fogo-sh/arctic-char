package main

import "core:math/linalg"
import sdl "vendor:sdl3"

PlayerCamera :: struct {
	position: [3]f32,
}

player_camera_update :: proc(
	camera: ^PlayerCamera,
	delta_time: f32,
	movement: Movement,
) -> matrix[4, 4]f32 {
	return linalg.MATRIX4F32_IDENTITY
}

player_camera_handle_mouse_button :: proc(
	camera: ^PlayerCamera,
	button: i32,
	pressed: bool,
	window: ^sdl.Window,
) -> bool {
	return false
}

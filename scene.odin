package main

import "core:math/linalg"

// The scene is intentionally tiny: one model rotating in front of one camera.
// Keeping this outside the renderer prevents GPU code from owning game choices.
scene_mvp :: proc(window_size: [2]i32, total_time: f32) -> matrix[4, 4]f32 {
	aspect := f32(window_size.x) / f32(window_size.y)
	model := linalg.matrix4_rotate_f32(total_time * 0.8, {0, 1, 0})
	view := linalg.matrix4_translate_f32({0, 0, -5})
	proj := linalg.matrix4_perspective_f32(linalg.to_radians(f32(70)), aspect, 0.1, 100)
	return proj * view * model
}

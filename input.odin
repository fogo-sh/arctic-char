package main

PlayerInput :: struct {
	move_forward: f32,
	move_right:   f32,
	jump_held:    bool,
	look_delta:   [2]f32,
}

input_begin_frame :: proc(input: ^PlayerInput) {
	input.look_delta = {}
}

package main

PlayerMoveInput :: struct {
	move_forward: f32,
	move_right:   f32,
	jump_held:    bool,
}

PlayerLookInput :: struct {
	look_delta:   [2]f32,
}

input_begin_frame :: proc(look: ^PlayerLookInput) {
	look.look_delta = {}
}

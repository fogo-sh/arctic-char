package engine

import sdl "vendor:sdl3"

INPUT_KEY_COUNT :: 512

InputState :: struct {
	keys:           [INPUT_KEY_COUNT]bool,
	mouse_delta:    [2]f32,
	mouse_captured: bool,
}

input_begin_frame :: proc(input: ^InputState) {
	input.mouse_delta = {}
}

input_key_down :: proc(input: InputState, scancode: sdl.Scancode) -> bool {
	index := int(scancode)
	return 0 <= index && index < len(input.keys) && input.keys[index]
}

package engine

Key :: enum int {
	W,
	A,
	S,
	D,
	Space,
	Escape,
	Q,
	COUNT,
}

INPUT_KEY_COUNT :: int(Key.COUNT)

InputState :: struct {
	keys:           [INPUT_KEY_COUNT]bool,
	mouse_delta:    [2]f32,
	mouse_captured: bool,
}

input_begin_frame :: proc(input: ^InputState) {
	input.mouse_delta = {}
}

input_key_down :: proc(input: InputState, key: Key) -> bool {
	index := int(key)
	return 0 <= index && index < len(input.keys) && input.keys[index]
}

input_set_key :: proc(input: ^InputState, key: Key, down: bool) {
	index := int(key)
	if 0 <= index && index < len(input.keys) {
		input.keys[index] = down
	}
}

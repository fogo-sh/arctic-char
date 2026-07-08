package engine

InputButton :: enum {
	W,
	A,
	S,
	D,
	Space,
}

InputState :: struct {
	buttons:    [InputButton]bool,
	mouse_delta: [2]f32,
}

input_begin_frame :: proc(input: ^InputState) {
	input.mouse_delta = {}
}

package engine

Game_API :: struct {
	init: proc(renderer: RendererApi, fs: ^GameFS, config: rawptr) -> rawptr,
	destroy: proc(game: rawptr),
	update: proc(game: rawptr, input: InputState, delta_time: f32),
	render: proc(game: rawptr, render_items: ^[dynamic]RenderItem, debug_lines: ^[dynamic]DebugLine, win_size: [2]i32) -> RenderFrame,
}

RenderFrame :: struct {
	globals: RenderPassGlobals,
	items: []RenderItem,
	debug_lines: []DebugLine,
	ui_items: []UiCommand,
	debug: DebugHudData,
}

FrameTimingStats :: struct {
	frame_ms: f32,
	update_ms: f32,
	render_ms: f32,
}

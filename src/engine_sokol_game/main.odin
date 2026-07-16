package main

import "base:runtime"
import "core:log"
import "core:os"
import "core:strings"

import engine "../engine"
import game "../game"
import engine_sokol "../engine_sokol"
import sapp "../../vendor/sokol/app"
import slog "../../vendor/sokol/log"

SOKOL_RENDER_ITEM_CAPACITY :: 256

Host :: struct {
	fs:           engine.GameFS,
	renderer:     engine_sokol.SokolRenderer,
	game:         rawptr,
	game_api:     engine.Game_API,
	input:        engine.InputState,
	ignore_next_mouse_motion: bool,
	render_items: [dynamic]engine.RenderItem,
	debug_lines:  [dynamic]engine.DebugLine,
	win_size:     [2]i32,
}

host: Host
custom_context: runtime.Context

init :: proc "c" () {
	context = custom_context
	engine.default_context = context

	args := os.args[1:]
	engine_config := engine.launch_config_parse(args)
	game_config := game.game_launch_config_parse(args)

	base_dir := engine_config.base_dir
	if base_dir == "" {
		base_dir = "."
	}
	host.fs = engine.game_fs_create(base_dir, engine_config.game)
	host.renderer = engine_sokol.create()
	host.render_items = make([dynamic]engine.RenderItem, 0, SOKOL_RENDER_ITEM_CAPACITY)
	host.debug_lines = make([dynamic]engine.DebugLine, 0, 4096)
	host.win_size = {i32(sapp.width()), i32(sapp.height())}
	host.game_api = game.game_api()
	host.game = host.game_api.init(engine_sokol.api(&host.renderer), &host.fs, &game_config)
	host.ignore_next_mouse_motion = true
	sapp.lock_mouse(true)
}

frame :: proc "c" () {
	context = custom_context
	host.input.mouse_captured = sapp.mouse_locked()
	host.win_size = {i32(sapp.width()), i32(sapp.height())}

	delta_time := f32(sapp.frame_duration())
	host.game_api.update(host.game, host.input, delta_time)
	frame := host.game_api.render(host.game, &host.render_items, &host.debug_lines, host.win_size)
	engine_sokol.draw(&host.renderer, frame, host.win_size)
	engine.input_begin_frame(&host.input)
}

event :: proc "c" (e: ^sapp.Event) {
	context = custom_context
	#partial switch e.type {
	case .KEY_DOWN:
		sokol_set_key(e.key_code, true)
		if e.key_code == .Q {
			sapp.quit()
		} else if e.key_code == .ESCAPE {
			sapp.lock_mouse(false)
			host.input.mouse_captured = false
			host.input.mouse_delta = {}
			host.ignore_next_mouse_motion = false
		}
	case .KEY_UP:
		sokol_set_key(e.key_code, false)
	case .MOUSE_DOWN:
		if !sapp.mouse_locked() {
			sapp.lock_mouse(true)
			host.input.mouse_delta = {}
			host.ignore_next_mouse_motion = true
		}
	case .MOUSE_MOVE:
		if sapp.mouse_locked() {
			if host.ignore_next_mouse_motion {
				host.ignore_next_mouse_motion = false
			} else {
				host.input.mouse_delta.x += e.mouse_dx
				host.input.mouse_delta.y += e.mouse_dy
			}
		}
	}
}

sokol_set_key :: proc(key_code: sapp.Keycode, down: bool) {
	#partial switch key_code {
	case .W:
		engine.input_set_key(&host.input, .W, down)
	case .A:
		engine.input_set_key(&host.input, .A, down)
	case .S:
		engine.input_set_key(&host.input, .S, down)
	case .D:
		engine.input_set_key(&host.input, .D, down)
	case .SPACE:
		engine.input_set_key(&host.input, .Space, down)
	case .ESCAPE:
		engine.input_set_key(&host.input, .Escape, down)
	case .Q:
		engine.input_set_key(&host.input, .Q, down)
	}
}

cleanup :: proc "c" () {
	context = custom_context
	if host.game_api.destroy != nil {
		host.game_api.destroy(host.game)
	}
	engine_sokol.destroy(&host.renderer)
	delete(host.debug_lines)
	delete(host.render_items)
	engine.game_fs_destroy(&host.fs)
	host = {}
}

main :: proc() {
	context.logger = log.create_console_logger()
	custom_context = context

	args := os.args[1:]
	config := engine.launch_config_parse(args)
	width := config.window_width
	if width <= 0 do width = 1024
	height := config.window_height
	if height <= 0 do height = 768
	title := config.window_title
	if title == "" do title = "arctic char* sokol"

	sapp.run({
		init_cb = init,
		frame_cb = frame,
		cleanup_cb = cleanup,
		event_cb = event,
		width = width,
		height = height,
		sample_count = 4,
		window_title = strings.clone_to_cstring(title, context.temp_allocator),
		icon = {sokol_default = true},
		logger = {func = slog.func},
	})
}

when ODIN_OS == .Windows {
	@(export)
	NvOptimusEnablement: u32 = 1

	@(export)
	AmdPowerXpressRequestHighPerformance: u32 = 1
}

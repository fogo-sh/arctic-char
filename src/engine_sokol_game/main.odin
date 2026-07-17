package main

import "base:runtime"
import "base:intrinsics"
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
	ui:           engine.UiContext,
	game:         rawptr,
	game_api:     engine.Game_API,
	input:        engine.InputState,
	ignore_next_mouse_motion: bool,
	render_items: [dynamic]engine.RenderItem,
	debug_lines:  [dynamic]engine.DebugLine,
	ui_commands:  [dynamic]engine.UiCommand,
	win_size:     [2]i32,
	draw_size:    [2]i32,
	dpi_scale:    f32,
	last_frame_ms: f32,
	last_update_ms: f32,
	last_render_ms: f32,
}

host: Host
custom_context: runtime.Context

when ODIN_OS == .Darwin {
	objc_id :: ^intrinsics.objc_object
	objc_SEL :: ^intrinsics.objc_selector

	NSPoint :: struct {
		x: f64,
		y: f64,
	}

	CGPoint :: struct {
		x: f64,
		y: f64,
	}

	CGSize :: struct {
		width:  f64,
		height: f64,
	}

	CGRect :: struct {
		origin: CGPoint,
		size:   CGSize,
	}

	foreign import ObjC "system:objc"
	@(default_calling_convention = "c")
	foreign ObjC {
		sel_registerName :: proc(name: cstring) -> objc_SEL ---
		objc_msgSend :: proc(self: objc_id, op: objc_SEL, #c_vararg args: ..any) ---
	}

	foreign import core_graphics "system:CoreGraphics.framework"
	@(default_calling_convention = "c")
	foreign core_graphics {
		CGMainDisplayID :: proc() -> u32 ---
		CGDisplayBounds :: proc(display: u32) -> CGRect ---
	}
}

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
	host.ui = engine.ui_create(i32(sapp.width()), i32(sapp.height()))
	host.render_items = make([dynamic]engine.RenderItem, 0, SOKOL_RENDER_ITEM_CAPACITY)
	host.debug_lines = make([dynamic]engine.DebugLine, 0, 4096)
	host.ui_commands = make([dynamic]engine.UiCommand, 0, engine.UI_COMMAND_CAPACITY)
	host.draw_size = {i32(sapp.width()), i32(sapp.height())}
	host.dpi_scale = sapp.dpi_scale()
	if host.dpi_scale <= 0 do host.dpi_scale = 1
	host.win_size = {i32(f32(host.draw_size.x) / host.dpi_scale), i32(f32(host.draw_size.y) / host.dpi_scale)}
	log.debugf("Sokol window size: logical=%dx%d pixels=%dx%d dpi_scale=%.2f high_dpi=%v", host.win_size.x, host.win_size.y, host.draw_size.x, host.draw_size.y, host.dpi_scale, sapp.high_dpi())
	host.game_api = game.game_api()
	host.game = host.game_api.init(engine_sokol.api(&host.renderer), &host.fs, &game_config)
	when ODIN_OS == .Darwin {
		sokol_apply_initial_window_position(engine_config)
	}
	host.ignore_next_mouse_motion = true
	sapp.lock_mouse(true)
}

when ODIN_OS == .Darwin {
	sokol_apply_initial_window_position :: proc(config: engine.LaunchConfig) {
		if !config.window_position_set do return
		window := sapp.macos_get_window()
		if window == nil do return

		display_bounds := CGDisplayBounds(CGMainDisplayID())
		top_left := NSPoint{
			x = f64(config.window_x),
			y = display_bounds.size.height - f64(config.window_y),
		}
		set_frame_top_left_point := sel_registerName("setFrameTopLeftPoint:")
		objc_msgSend(cast(objc_id)window, set_frame_top_left_point, top_left)
	}
}

frame :: proc "c" () {
	context = custom_context
	frame_start := engine.performance_counter_now()
	host.input.mouse_captured = sapp.mouse_locked()
	host.draw_size = {i32(sapp.width()), i32(sapp.height())}
	host.dpi_scale = sapp.dpi_scale()
	if host.dpi_scale <= 0 do host.dpi_scale = 1
	host.win_size = {i32(f32(host.draw_size.x) / host.dpi_scale), i32(f32(host.draw_size.y) / host.dpi_scale)}

	delta_time := f32(sapp.frame_duration())
	update_start := engine.performance_counter_now()
	host.game_api.update(host.game, host.input, delta_time)
	host.last_update_ms = engine.performance_elapsed_ms(update_start)
	render_start := engine.performance_counter_now()
	frame := host.game_api.render(host.game, &host.render_items, &host.debug_lines, host.draw_size)
	clear(&host.ui_commands)
	for item in frame.ui_items {
		append(&host.ui_commands, item)
	}
	engine.ui_debug_hud_append_commands(&host.ui, host.win_size, frame.debug, {
		frame_ms = host.last_frame_ms,
		update_ms = host.last_update_ms,
		render_ms = host.last_render_ms,
	}, &host.ui_commands)
	engine_sokol.draw(&host.renderer, frame, host.draw_size, host.ui_commands[:], {host.dpi_scale, host.dpi_scale})
	host.last_render_ms = engine.performance_elapsed_ms(render_start)
	engine.input_begin_frame(&host.input)
	host.last_frame_ms = engine.performance_elapsed_ms(frame_start)
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
	engine.ui_destroy(&host.ui)
	delete(host.ui_commands)
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
		high_dpi = true,
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

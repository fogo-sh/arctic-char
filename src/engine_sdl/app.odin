package engine_sdl

import "base:runtime"
import "core:c"
import "core:log"
import "core:strings"
import engine "../engine"
import sdl "vendor:sdl3"

default_context: runtime.Context

APP_RENDER_ITEM_CAPACITY :: 256

App :: struct {
	window: ^sdl.Window,
	gpu:    ^sdl.GPUDevice,

	renderer: Renderer,
	ui:       UiContext,
	input: InputState,
	render_items: [dynamic]RenderItem,
	debug_lines:  [dynamic]DebugLine,
	ui_commands:  [dynamic]UiCommand,
	fs: GameFS,
	game: rawptr,
	game_api: Game_API,
	win_size: [2]i32,
	draw_size: [2]i32,

	running:    bool,
	mouse_captured: bool,
	ignore_next_mouse_motion: bool,
	last_ticks: u64,
	total_time: f32,
	stats_log_time: f32,
	last_frame_ms: f32,
	last_update_ms: f32,
	last_render_ms: f32,
}

// Creates SDL's application shell: window, GPU device, and the first renderer.
// Asset loading is still explicit here so startup order is easy to follow.
app_create :: proc(config: LaunchConfig) -> App {
	when ODIN_DEBUG {
		_ = sdl.SetHint(sdl.HINT_RENDER_GPU_DEBUG, "1")
	}

	sdl.SetLogPriorities(.VERBOSE)
	sdl.SetLogOutputFunction(
		proc "c" (
			userdata: rawptr,
			category: sdl.LogCategory,
			priority: sdl.LogPriority,
			message: cstring,
		) {
			context = default_context
			log.debugf("SDL {} [{}]: {}", category, priority, message)
		},
		nil,
	)

	ok := sdl.SetAppMetadata("arctic char*", "0.1.0", "sh.fogo.arctic-char")
	assert(ok)

	ok = sdl.Init({.VIDEO})
	assert(ok)

	app := App{running = true, mouse_captured = true, ignore_next_mouse_motion = true}

	window_flags := sdl.WindowFlags{.RESIZABLE, .HIGH_PIXEL_DENSITY}
	if config.fullscreen {
		window_flags += sdl.WindowFlags{.FULLSCREEN}
	}
	window_title := config.window_title
	if window_title == "" {
		window_title = "arctic char*"
	}
	window_width := config.window_width
	if window_width <= 0 {
		window_width = 1024
	}
	window_height := config.window_height
	if window_height <= 0 {
		window_height = 768
	}
	app.window = sdl.CreateWindow(strings.clone_to_cstring(window_title, context.temp_allocator), window_width, window_height, window_flags)
	assert(app.window != nil)
	if config.window_position_set {
		ok = sdl.SetWindowPosition(app.window, c.int(config.window_x), c.int(config.window_y))
		assert(ok)
	}
	ok = sdl.SetWindowRelativeMouseMode(app.window, true)
	assert(ok)

	gpu_debug := ODIN_DEBUG
	app.gpu = sdl.CreateGPUDevice(shader_format, gpu_debug, nil)
	assert(app.gpu != nil)

	ok = sdl.ClaimWindowForGPUDevice(app.gpu, app.window)
	assert(ok)

	ok = sdl.GetWindowSize(app.window, &app.win_size.x, &app.win_size.y)
	assert(ok)
	app.draw_size = app_sdl_pixel_size(app.window, app.win_size)
	app_log_sdl_dpi(app.window, app.win_size, app.draw_size)

	base_dir := app_resolve_base_dir(config.base_dir)
	app.fs = game_fs_create(base_dir, config.game)
	app.renderer = renderer_create(app.gpu, app.window, app.draw_size.x, app.draw_size.y, config.sdl_msaa_samples)
	app.ui = ui_create(app.win_size.x, app.win_size.y)
	app.render_items = make([dynamic]RenderItem, 0, APP_RENDER_ITEM_CAPACITY)
	app.debug_lines = make([dynamic]DebugLine, 0, 4096)
	app.ui_commands = make([dynamic]UiCommand, 0, UI_COMMAND_CAPACITY)

	app.last_ticks = sdl.GetTicks()
	return app
}

app_resolve_base_dir :: proc(config_base_dir: string) -> string {
	if config_base_dir != "" {
		return config_base_dir
	}

	when ODIN_OS == .Darwin {
		base_path_c := sdl.GetBasePath()
		if base_path_c != nil {
			base_path := string(base_path_c)
			if strings.contains(base_path, ".app/Contents/Resources") {
				return base_path
			}
		}
	}

	return "."
}

app_sdl_pixel_size :: proc(window: ^sdl.Window, fallback: [2]i32) -> [2]i32 {
	result := fallback
	ok := sdl.GetWindowSizeInPixels(window, &result.x, &result.y)
	if !ok || result.x <= 0 || result.y <= 0 {
		return fallback
	}
	return result
}

app_log_sdl_dpi :: proc(window: ^sdl.Window, logical, pixels: [2]i32) {
	pixel_density := sdl.GetWindowPixelDensity(window)
	display_scale := sdl.GetWindowDisplayScale(window)
	log.debugf("SDL window size: logical=%dx%d pixels=%dx%d pixel_density=%.2f display_scale=%.2f", logical.x, logical.y, pixels.x, pixels.y, pixel_density, display_scale)
}

app_ui_scale :: proc(app: ^App) -> [2]f32 {
	if app.win_size.x <= 0 || app.win_size.y <= 0 do return {1, 1}
	return {f32(app.draw_size.x) / f32(app.win_size.x), f32(app.draw_size.y) / f32(app.win_size.y)}
}

app_init_game :: proc(app: ^App, game_api: Game_API, game_config: rawptr) {
	app.game_api = game_api
	app.game = app.game_api.init(renderer_sdlgpu_api(&app.renderer), &app.fs, game_config)
}

app_destroy :: proc(app: ^App) {
	if app.game_api.destroy != nil {
		app.game_api.destroy(app.game)
	}
	ui_destroy(&app.ui)
	renderer_destroy(&app.renderer)
	delete(app.ui_commands)
	delete(app.debug_lines)
	delete(app.render_items)
	game_fs_destroy(&app.fs)
	if app.gpu != nil && app.window != nil do sdl.ReleaseWindowFromGPUDevice(app.gpu, app.window)
	if app.gpu != nil do sdl.DestroyGPUDevice(app.gpu)
	if app.window != nil do sdl.DestroyWindow(app.window)
	sdl.Quit()
	app^ = {}
}

app_run :: proc(app: ^App) {
	for app.running {
		app_frame(app)
	}

	log.debug("Goodbye!")
}

app_should_run :: proc(app: ^App) -> bool {
	return app.running
}

app_frame :: proc(app: ^App) {
	frame_start := performance_counter_now()
	input_begin_frame(&app.input)
	app.input.mouse_captured = app.mouse_captured
	app_handle_events(app)
	app_tick(app)
	app_draw(app)
	app.last_frame_ms = performance_elapsed_ms(frame_start)
}

app_tick :: proc(app: ^App) {
	new_ticks := sdl.GetTicks()
	delta_time := f32(new_ticks - app.last_ticks) / 1000
	app.last_ticks = new_ticks
	app.total_time += delta_time
	app_update_input(app)
	update_start := performance_counter_now()
	app.game_api.update(app.game, app.input, delta_time)
	app.last_update_ms = performance_elapsed_ms(update_start)
	app.stats_log_time += delta_time
}

app_update_input :: proc(app: ^App) {
	num_keys: c.int
	keys := sdl.GetKeyboardState(&num_keys)
	input_set_key(&app.input, .W, int(sdl.Scancode.W) < int(num_keys) && keys[sdl.Scancode.W])
	input_set_key(&app.input, .A, int(sdl.Scancode.A) < int(num_keys) && keys[sdl.Scancode.A])
	input_set_key(&app.input, .S, int(sdl.Scancode.S) < int(num_keys) && keys[sdl.Scancode.S])
	input_set_key(&app.input, .D, int(sdl.Scancode.D) < int(num_keys) && keys[sdl.Scancode.D])
	input_set_key(&app.input, .Space, int(sdl.Scancode.SPACE) < int(num_keys) && keys[sdl.Scancode.SPACE])
	input_set_key(&app.input, .Escape, int(sdl.Scancode.ESCAPE) < int(num_keys) && keys[sdl.Scancode.ESCAPE])
	input_set_key(&app.input, .Q, int(sdl.Scancode.Q) < int(num_keys) && keys[sdl.Scancode.Q])
}

app_handle_events :: proc(app: ^App) {
	ev: sdl.Event
	for sdl.PollEvent(&ev) {
		#partial switch ev.type {
		case .QUIT:
			app.running = false
		case .WINDOW_RESIZED, .WINDOW_PIXEL_SIZE_CHANGED, .WINDOW_DISPLAY_SCALE_CHANGED:
			ok := sdl.GetWindowSize(app.window, &app.win_size.x, &app.win_size.y)
			assert(ok)
			app.draw_size = app_sdl_pixel_size(app.window, app.win_size)
			app_log_sdl_dpi(app.window, app.win_size, app.draw_size)
			renderer_resize(&app.renderer, app.draw_size.x, app.draw_size.y)
		case .MOUSE_MOTION:
			if app.mouse_captured {
				if app.ignore_next_mouse_motion {
					app.ignore_next_mouse_motion = false
				} else {
					app.input.mouse_delta.x += ev.motion.xrel
					app.input.mouse_delta.y += ev.motion.yrel
				}
			}
		case .MOUSE_BUTTON_DOWN:
			if !app.mouse_captured {
				ok := sdl.SetWindowRelativeMouseMode(app.window, true)
				assert(ok)
				app.mouse_captured = true
				app.input.mouse_captured = true
				app.input.mouse_delta = {}
				app.ignore_next_mouse_motion = true
			}
		case .KEY_DOWN:
			if ev.key.scancode == .Q {
				app.running = false
			} else if ev.key.scancode == .ESCAPE {
				ok := sdl.SetWindowRelativeMouseMode(app.window, false)
				assert(ok)
				app.mouse_captured = false
				app.input.mouse_captured = false
				app.input.mouse_delta = {}
				app.ignore_next_mouse_motion = false
			}
		}
	}
}

app_draw :: proc(app: ^App) {
	render_start := performance_counter_now()
	cmd_buf := sdl.AcquireGPUCommandBuffer(app.gpu)
	swapchain_tex: ^sdl.GPUTexture
	swapchain_width, swapchain_height: u32
	ok := sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, app.window, &swapchain_tex, &swapchain_width, &swapchain_height)
	assert(ok)
	if swapchain_width > 0 && swapchain_height > 0 {
		new_draw_size := [2]i32{i32(swapchain_width), i32(swapchain_height)}
		if new_draw_size != app.draw_size {
			log.debugf("SDL draw size changed: logical=%dx%d draw=%dx%d", app.win_size.x, app.win_size.y, new_draw_size.x, new_draw_size.y)
			app.draw_size = new_draw_size
			renderer_resize(&app.renderer, app.draw_size.x, app.draw_size.y)
		}
	}

	frame := app.game_api.render(app.game, &app.render_items, &app.debug_lines, app.draw_size)
	clear(&app.ui_commands)
	for item in frame.ui_items {
		append(&app.ui_commands, item)
	}
	ui_debug_hud_append_commands(&app.ui, app.win_size, frame.debug, {
		frame_ms = app.last_frame_ms,
		update_ms = app.last_update_ms,
		render_ms = app.last_render_ms,
	}, &app.ui_commands)
	renderer_draw(&app.renderer, cmd_buf, swapchain_tex, frame.globals, frame.items, frame.debug_lines, app.ui_commands[:], app.draw_size, app_ui_scale(app))
	if app.stats_log_time >= 2.0 {
		renderer_log_stats(&app.renderer)
		app.stats_log_time = 0
	}

	ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
	assert(ok)
	app.last_render_ms = performance_elapsed_ms(render_start)
}

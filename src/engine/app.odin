package engine

import "base:runtime"
import "core:c"
import "core:log"
import "core:strings"
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
	ui_commands:  [dynamic]UiCommand,
	fs: GameFS,
	game: rawptr,
	game_api: Game_API,
	win_size: [2]i32,

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

Game_API :: struct {
	init: proc(renderer: ^Renderer, fs: ^GameFS, config: rawptr) -> rawptr,
	destroy: proc(game: rawptr),
	update: proc(game: rawptr, input: InputState, delta_time: f32),
	render: proc(game: rawptr, render_items: ^[dynamic]RenderItem, win_size: [2]i32) -> RenderFrame,
}

RenderFrame :: struct {
	globals: RenderPassGlobals,
	items: []RenderItem,
	ui_items: []UiCommand,
	debug: DebugHudData,
}

FrameTimingStats :: struct {
	frame_ms: f32,
	update_ms: f32,
	render_ms: f32,
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

	window_flags := sdl.WindowFlags{.RESIZABLE}
	if config.fullscreen {
		window_flags += sdl.WindowFlags{.FULLSCREEN}
	}
	app.window = sdl.CreateWindow("arctic char*", 1024, 768, window_flags)
	assert(app.window != nil)
	ok = sdl.SetWindowRelativeMouseMode(app.window, true)
	assert(ok)

	gpu_debug := ODIN_DEBUG
	app.gpu = sdl.CreateGPUDevice(shader_format, gpu_debug, nil)
	assert(app.gpu != nil)

	ok = sdl.ClaimWindowForGPUDevice(app.gpu, app.window)
	assert(ok)

	ok = sdl.GetWindowSize(app.window, &app.win_size.x, &app.win_size.y)
	assert(ok)

	base_dir := app_resolve_base_dir(config.base_dir)
	app.fs = game_fs_create(base_dir, config.game)
	app.renderer = renderer_create(app.gpu, app.window, app.win_size.x, app.win_size.y)
	app.ui = ui_create(app.win_size.x, app.win_size.y)
	app.render_items = make([dynamic]RenderItem, 0, APP_RENDER_ITEM_CAPACITY)
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

app_init_game :: proc(app: ^App, game_api: Game_API, game_config: rawptr) {
	app.game_api = game_api
	app.game = app.game_api.init(&app.renderer, &app.fs, game_config)
}

app_destroy :: proc(app: ^App) {
	if app.game_api.destroy != nil {
		app.game_api.destroy(app.game)
	}
	ui_destroy(&app.ui)
	renderer_destroy(&app.renderer)
	delete(app.ui_commands)
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
	frame_start := sdl.GetPerformanceCounter()
	input_begin_frame(&app.input)
	app.input.mouse_captured = app.mouse_captured
	app_handle_events(app)
	app_tick(app)
	app_draw(app)
	app.last_frame_ms = app_elapsed_ms(frame_start)
}

app_tick :: proc(app: ^App) {
	new_ticks := sdl.GetTicks()
	delta_time := f32(new_ticks - app.last_ticks) / 1000
	app.last_ticks = new_ticks
	app.total_time += delta_time
	app_update_input(app)
	update_start := sdl.GetPerformanceCounter()
	app.game_api.update(app.game, app.input, delta_time)
	app.last_update_ms = app_elapsed_ms(update_start)
	app.stats_log_time += delta_time
}

app_update_input :: proc(app: ^App) {
	num_keys: c.int
	keys := sdl.GetKeyboardState(&num_keys)
	key_count := min(len(app.input.keys), int(num_keys))
	for i in 0..<key_count {
		app.input.keys[i] = keys[i]
	}
}

app_handle_events :: proc(app: ^App) {
	ev: sdl.Event
	for sdl.PollEvent(&ev) {
		#partial switch ev.type {
		case .QUIT:
			app.running = false
		case .WINDOW_RESIZED:
			ok := sdl.GetWindowSize(app.window, &app.win_size.x, &app.win_size.y)
			assert(ok)
			renderer_resize(&app.renderer, app.win_size.x, app.win_size.y)
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
	render_start := sdl.GetPerformanceCounter()
	cmd_buf := sdl.AcquireGPUCommandBuffer(app.gpu)
	swapchain_tex: ^sdl.GPUTexture
	ok := sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, app.window, &swapchain_tex, nil, nil)
	assert(ok)

	frame := app.game_api.render(app.game, &app.render_items, app.win_size)
	clear(&app.ui_commands)
	for item in frame.ui_items {
		append(&app.ui_commands, item)
	}
	ui_debug_hud_append_commands(&app.ui, app.win_size, frame.debug, {
		frame_ms = app.last_frame_ms,
		update_ms = app.last_update_ms,
		render_ms = app.last_render_ms,
	}, &app.ui_commands)
	renderer_draw(&app.renderer, cmd_buf, swapchain_tex, frame.globals, frame.items, app.ui_commands[:], app.win_size)
	if app.stats_log_time >= 2.0 {
		renderer_log_stats(&app.renderer)
		app.stats_log_time = 0
	}

	ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
	assert(ok)
	app.last_render_ms = app_elapsed_ms(render_start)
}

app_elapsed_ms :: proc(start_counter: u64) -> f32 {
	end_counter := sdl.GetPerformanceCounter()
	frequency := sdl.GetPerformanceFrequency()
	return f32(end_counter - start_counter) * 1000.0 / f32(frequency)
}

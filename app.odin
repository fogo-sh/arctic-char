package main

import "base:runtime"
import "core:log"
import sdl "vendor:sdl3"

default_context: runtime.Context

App :: struct {
	window: ^sdl.Window,
	gpu:    ^sdl.GPUDevice,

	renderer: Renderer,
	scene:    Scene,
	render_items: [dynamic]RenderItem,
	win_size: [2]i32,

	running:    bool,
	last_ticks: u64,
	total_time: f32,
}

// Creates SDL's application shell: window, GPU device, and the first renderer.
// Asset loading is still explicit here so startup order is easy to follow.
app_create :: proc() -> App {
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

	app := App{running = true}

	app.window = sdl.CreateWindow("arctic char*", 1024, 768, {.RESIZABLE})
	assert(app.window != nil)

	gpu_debug := ODIN_DEBUG
	app.gpu = sdl.CreateGPUDevice(shader_format, gpu_debug, nil)
	assert(app.gpu != nil)

	ok = sdl.ClaimWindowForGPUDevice(app.gpu, app.window)
	assert(ok)

	ok = sdl.GetWindowSize(app.window, &app.win_size.x, &app.win_size.y)
	assert(ok)

	assets := scene_assets_load()
	app.renderer = renderer_create(app.gpu, app.window, app.win_size.x, app.win_size.y, assets.render_meshes[:])
	app.scene = scene_create(&assets)
	app.render_items = make([dynamic]RenderItem, 0, MAX_SUZANNES + 1)
	scene_assets_destroy(&assets)

	app.last_ticks = sdl.GetTicks()
	return app
}

app_destroy :: proc(app: ^App) {
	scene_destroy(&app.scene)
	renderer_destroy(&app.renderer)
	delete(app.render_items)
	if app.gpu != nil && app.window != nil do sdl.ReleaseWindowFromGPUDevice(app.gpu, app.window)
	if app.gpu != nil do sdl.DestroyGPUDevice(app.gpu)
	if app.window != nil do sdl.DestroyWindow(app.window)
	sdl.Quit()
	app^ = {}
}

app_run :: proc(app: ^App) {
	for app.running {
		app_update_time(app)
		app_handle_events(app)
		app_draw(app)
	}

	log.debug("Goodbye!")
}

app_update_time :: proc(app: ^App) {
	new_ticks := sdl.GetTicks()
	delta_time := f32(new_ticks - app.last_ticks) / 1000
	app.last_ticks = new_ticks
	app.total_time += delta_time
	scene_update(&app.scene, delta_time)
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
		case .KEY_DOWN:
			if ev.key.scancode == .Q || ev.key.scancode == .ESCAPE {
				app.running = false
			}
		}
	}
}

app_draw :: proc(app: ^App) {
	cmd_buf := sdl.AcquireGPUCommandBuffer(app.gpu)
	swapchain_tex: ^sdl.GPUTexture
	ok := sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, app.window, &swapchain_tex, nil, nil)
	assert(ok)

	render_items := scene_collect_render_items(&app.scene, app.win_size, &app.render_items)
	renderer_draw(&app.renderer, cmd_buf, swapchain_tex, render_items)

	ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
	assert(ok)
}

package game

import "core:log"
import "core:time"
import engine "../engine"

Run :: proc() {
	context.logger = log.create_console_logger()
	engine.default_context = context

	config := engine.launch_config_parse()
	app := engine.app_create(config)
	defer engine.app_destroy(&app)
	engine.app_init_game(&app, config, game_api())

	engine.app_run(&app)
}

Game_State :: struct {
	renderer: ^Renderer,
	fs: ^GameFS,
	map_qpath: string,
	map_mtime: time.Time,
	reload_check_timer: f32,
	scene: Scene,
}

game_api :: proc() -> engine.Game_API {
	return {
		init = game_init,
		destroy = game_destroy,
		update = game_update,
		render = game_render,
	}
}

game_init :: proc(renderer: ^Renderer, fs: ^GameFS, config: LaunchConfig) -> rawptr {
	state := new(Game_State)
	state.renderer = renderer
	state.fs = fs
	state.map_qpath = launch_config_map_qpath(config)
	state.map_mtime, _ = game_fs_modification_time(fs, state.map_qpath)
	assets := scene_assets_load(fs, config)
	upload := renderer_begin_upload(renderer)
	assets.suzanne_handle = renderer_upload_mesh(&upload, &assets.suzanne_mesh)
	assets.map_handle = renderer_upload_mesh(&upload, &assets.level.render_mesh)
	renderer_end_upload(&upload)
	assets.default_material = renderer_default_material()
	state.scene = scene_create(&assets)
	scene_assets_destroy(&assets)
	return state
}

game_destroy :: proc(game: rawptr) {
	state := cast(^Game_State)game
	scene_destroy(&state.scene)
	delete(state.map_qpath)
	free(state)
}

game_update :: proc(game: rawptr, move_input: PlayerMoveInput, look_input: PlayerLookInput, delta_time: f32) {
	state := cast(^Game_State)game
	game_reload_map_if_changed(state, delta_time)
	scene_update(&state.scene, move_input, look_input, delta_time)
}

game_render :: proc(game: rawptr, render_items: ^[dynamic]RenderItem, win_size: [2]i32) -> engine.RenderFrame {
	state := cast(^Game_State)game
	return {
		globals = scene_render_globals(&state.scene, win_size),
		items = scene_collect_render_items(&state.scene, render_items),
	}
}

game_reload_map_if_changed :: proc(state: ^Game_State, delta_time: f32) {
	state.reload_check_timer += delta_time
	if state.reload_check_timer < 0.25 {
		return
	}
	state.reload_check_timer = 0

	mtime, ok := game_fs_modification_time(state.fs, state.map_qpath)
	if !ok || mtime == state.map_mtime {
		return
	}

	log.debugf("Reloading map: %s", state.map_qpath)
	level := level_load(state.fs, state.map_qpath)
	renderer_replace_mesh(state.renderer, state.scene.map_mesh, &level.render_mesh)
	scene_reload_level(&state.scene, &level)
	level_destroy(&level)
	state.map_mtime = mtime
}

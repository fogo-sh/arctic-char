package game

import "core:log"
import "core:os"
import "core:strings"
import "core:time"
import engine "../engine"

Run :: proc() {
	context.logger = log.create_console_logger()
	engine.default_context = context

	args := os.args[1:]
	engine_config := engine.launch_config_parse(args)
	game_config := game_launch_config_parse(args)
	app := engine.app_create(engine_config)
	defer engine.app_destroy(&app)
	engine.app_init_game(&app, game_api(), &game_config)

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

g: ^Game_State

GameLaunchConfig :: struct {
	map_name: string,
}

game_api :: proc() -> engine.Game_API {
	return {
		init = game_init,
		destroy = game_destroy,
		update = game_update,
		render = game_render,
	}
}

@(export)
game_init :: proc(renderer: ^Renderer, fs: ^GameFS, config: rawptr) -> rawptr {
	state := new(Game_State)
	g = state
	state.renderer = renderer
	state.fs = fs
	game_config := (cast(^GameLaunchConfig)config)^
	state.map_qpath = game_launch_config_map_qpath(game_config)
	state.map_mtime, _ = engine.game_fs_modification_time(fs, state.map_qpath)
	assets := scene_assets_load(fs, state.map_qpath)
	gpu_resources := scene_gpu_resources_upload(renderer, &assets)
	state.scene = scene_create(&assets, gpu_resources)
	scene_assets_destroy(&assets)
	return state
}

@(export)
game_destroy :: proc(game: rawptr) {
	state := cast(^Game_State)game
	scene_destroy(&state.scene)
	delete(state.map_qpath)
	free(state)
	if g == state {
		g = nil
	}
}

@(export)
game_update :: proc(game: rawptr, input: InputState, delta_time: f32) {
	state := cast(^Game_State)game
	game_reload_map_if_changed(state, delta_time)
	move_input, look_input := player_input_from_engine(input)
	scene_update(&state.scene, move_input, look_input, delta_time)
}

@(export)
game_render :: proc(game: rawptr, render_items: ^[dynamic]RenderItem, win_size: [2]i32) -> engine.RenderFrame {
	state := cast(^Game_State)game
	return {
		globals = scene_render_globals(&state.scene, win_size),
		items = scene_collect_render_items(&state.scene, render_items),
	}
}

@(export)
game_memory :: proc() -> rawptr {
	return g
}

@(export)
game_memory_size :: proc() -> int {
	return size_of(Game_State)
}

@(export)
game_before_hot_reload :: proc(game: rawptr) {
	state := cast(^Game_State)game
	scene_prepare_hot_reload(&state.scene)
}

@(export)
game_hot_reloaded :: proc(mem: rawptr) {
	g = cast(^Game_State)mem
	scene_rebuild_after_hot_reload(&g.scene, g.fs, g.map_qpath)
}

@(export)
game_force_restart :: proc() -> bool {
	return false
}

game_launch_config_parse :: proc(args: []string) -> GameLaunchConfig {
	config := GameLaunchConfig{map_name = "test"}
	for i := 0; i < len(args); i += 1 {
		if args[i] == "+map" && i + 1 < len(args) {
			i += 1
			config.map_name = args[i]
		}
	}
	return config
}

game_launch_config_map_qpath :: proc(config: GameLaunchConfig, allocator := context.allocator) -> string {
	// Make `+map test` resolve to `maps/test.map`.
	qpath, err := strings.concatenate({"maps/", config.map_name, ".map"}, allocator)
	assert(err == nil)
	return qpath
}

game_reload_map_if_changed :: proc(state: ^Game_State, delta_time: f32) {
	state.reload_check_timer += delta_time
	if state.reload_check_timer < 0.25 {
		return
	}
	state.reload_check_timer = 0

	mtime, ok := engine.game_fs_modification_time(state.fs, state.map_qpath)
	if !ok || mtime == state.map_mtime {
		return
	}

	log.debugf("Reloading map: %s", state.map_qpath)
	level := level_load(state.fs, state.map_qpath)
	engine.renderer_replace_mesh(state.renderer, state.scene.map_mesh, &level.render_mesh)
	scene_reload_level(&state.scene, &level)
	level_destroy(&level)
	state.map_mtime = mtime
}

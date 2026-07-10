package main

import "core:dynlib"
import "core:fmt"
import flags "core:flags"
import "core:log"
import "core:os"
import "core:time"
import engine "../engine"

when ODIN_OS == .Windows {
	DLL_EXT :: ".dll"
} else when ODIN_OS == .Darwin {
	DLL_EXT :: ".dylib"
} else {
	DLL_EXT :: ".so"
}

GAME_DLL_DIR :: "build/hot_reload/"
GAME_DLL_PATH :: GAME_DLL_DIR + "game" + DLL_EXT

Hot_Game_API :: struct {
	lib: dynlib.Library,
	init: proc(renderer: ^engine.Renderer, fs: ^engine.GameFS, config: rawptr) -> rawptr,
	destroy: proc(game: rawptr),
	update: proc(game: rawptr, input: engine.InputState, delta_time: f32),
	render: proc(game: rawptr, render_items: ^[dynamic]engine.RenderItem, win_size: [2]i32) -> engine.RenderFrame,
	memory: proc() -> rawptr,
	memory_size: proc() -> int,
	before_hot_reload: proc(game: rawptr),
	hot_reloaded: proc(mem: rawptr),
	force_restart: proc() -> bool,
	modification_time: time.Time,
	api_version: int,
}

main :: proc() {
	context.logger = log.create_console_logger()
	engine.default_context = context

	args := os.args[1:]
	engine_config := engine.launch_config_parse(args)
	game_config := game_launch_config_parse(args)

	api_version := 0
	game_api, ok := hot_load_game_api(api_version)
	if !ok {
		return
	}
	api_version += 1

	app := engine.app_create(engine_config)
	engine.app_init_game(&app, hot_engine_api(game_api), &game_config)
	old_game_apis := make([dynamic]Hot_Game_API)

	defer {
		for &old_api in old_game_apis {
			hot_unload_game_api(&old_api)
		}
		delete(old_game_apis)
		hot_unload_game_api(&game_api)
	}
	defer engine.app_destroy(&app)

	for engine.app_should_run(&app) {
		force_restart := game_api.force_restart != nil && game_api.force_restart()
		reload := force_restart || hot_game_dylib_changed(game_api)
		if reload {
			new_api, loaded := hot_load_game_api(api_version)
			if loaded {
				api_version += 1
				force_restart = force_restart || game_api.memory_size() != new_api.memory_size()
				if force_restart {
					log.debug("Game dylib changed; full app restart required")
					engine.app_destroy(&app)
					for &old_api in old_game_apis {
						hot_unload_game_api(&old_api)
					}
					clear(&old_game_apis)
					hot_unload_game_api(&game_api)
					game_api = new_api
					app = engine.app_create(engine_config)
					engine.app_init_game(&app, hot_engine_api(game_api), &game_config)
				} else {
					log.debug("Game dylib changed; preserving game state and rebuilding physics")
					game_api.before_hot_reload(app.game)
					append(&old_game_apis, game_api)
					game_api = new_api
					app.game_api = hot_engine_api(game_api)
					game_api.hot_reloaded(app.game)
				}
			}
		}

		engine.app_frame(&app)
	}

	log.debug("Goodbye!")
}

hot_engine_api :: proc(api: Hot_Game_API) -> engine.Game_API {
	return {
		init = api.init,
		destroy = api.destroy,
		update = api.update,
		render = api.render,
	}
}

hot_load_game_api :: proc(api_version: int) -> (api: Hot_Game_API, ok: bool) {
	mod_time, mod_time_error := os.last_write_time_by_name(GAME_DLL_PATH)
	if mod_time_error != os.ERROR_NONE {
		log.errorf("Failed getting modification time for %s: %v", GAME_DLL_PATH, mod_time_error)
		return
	}

	game_dll_name := fmt.tprintf(GAME_DLL_DIR + "game_%d" + DLL_EXT, api_version)
	copy_err := os.copy_file(game_dll_name, GAME_DLL_PATH)
	if copy_err != nil {
		log.errorf("Failed copying %s to %s: %v", GAME_DLL_PATH, game_dll_name, copy_err)
		return
	}

	_, ok = dynlib.initialize_symbols(&api, game_dll_name, "game_", "lib")
	if !ok {
		log.errorf("Failed initializing game dylib symbols: %s", dynlib.last_error())
		return
	}

	api.modification_time = mod_time
	api.api_version = api_version
	return
}

hot_unload_game_api :: proc(api: ^Hot_Game_API) {
	if api.lib != nil && !dynlib.unload_library(api.lib) {
		log.errorf("Failed unloading game dylib: %s", dynlib.last_error())
	}
	api^ = {}
}

hot_game_dylib_changed :: proc(api: Hot_Game_API) -> bool {
	mod_time, mod_time_error := os.last_write_time_by_name(GAME_DLL_PATH)
	return mod_time_error == os.ERROR_NONE && mod_time != api.modification_time
}

GameLaunchConfig :: struct {
	map_name: string,
}

GameLaunchOptions :: struct {
	basedir: string `usage:"Base directory consumed by engine code."`,
	game:    string `usage:"Game directory consumed by engine code."`,
	map_name: string `args:"name=map" usage:"Map name to load from maps/<name>.map."`,
}

game_launch_config_parse :: proc(args: []string) -> GameLaunchConfig {
	config := GameLaunchConfig{map_name = "test"}
	options := game_launch_options_parse(args)
	if options.map_name != "" {
		config.map_name = options.map_name
	}
	return config
}

game_launch_options_parse :: proc(args: []string) -> GameLaunchOptions {
	runtime_args := make([dynamic]string, 0, len(args) + 1, context.temp_allocator)
	append(&runtime_args, "arctic-char-hot")
	for arg in args {
		append(&runtime_args, arg)
	}

	options: GameLaunchOptions
	flags.parse_or_exit(&options, runtime_args[:], .Unix)
	return options
}

when ODIN_OS == .Windows {
	@(export)
	NvOptimusEnablement: u32 = 1

	@(export)
	AmdPowerXpressRequestHighPerformance: u32 = 1
}

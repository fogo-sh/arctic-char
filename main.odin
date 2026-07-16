package main

import "core:log"
import "core:os"
import engine "./src/engine"
import engine_sdl "./src/engine_sdl"
import game "./src/game"

main :: proc() {
	context.logger = log.create_console_logger()
	engine.default_context = context
	engine_sdl.default_context = context

	args := os.args[1:]
	engine_config := engine.launch_config_parse(args)
	game_config := game.game_launch_config_parse(args)
	app := engine_sdl.app_create(engine_config)
	defer engine_sdl.app_destroy(&app)
	engine_sdl.app_init_game(&app, game.game_api(), &game_config)

	engine_sdl.app_run(&app)
}

when ODIN_OS == .Windows {
	@(export)
	NvOptimusEnablement: u32 = 1

	@(export)
	AmdPowerXpressRequestHighPerformance: u32 = 1
}

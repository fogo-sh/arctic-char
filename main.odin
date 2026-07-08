package main

import "core:log"

main :: proc() {
	context.logger = log.create_console_logger()
	default_context = context

	config := launch_config_parse()
	app := app_create(config)
	defer app_destroy(&app)

	app_run(&app)
}

@(export)
NvOptimusEnablement: u32 = 1

@(export)
AmdPowerXpressRequestHighPerformance: i32 = 1

package main

import "core:log"

main :: proc() {
	context.logger = log.create_console_logger()
	default_context = context

	app := app_create()
	defer app_destroy(&app)

	app_run(&app)
}

@(export)
NvOptimusEnablement: u32 = 1

@(export)
AmdPowerXpressRequestHighPerformance: i32 = 1

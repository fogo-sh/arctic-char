package game

import "core:log"

Run :: proc() {
	context.logger = log.create_console_logger()
	default_context = context

	config := launch_config_parse()
	app := app_create(config)
	defer app_destroy(&app)

	app_run(&app)
}

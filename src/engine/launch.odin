package engine

import flags "core:flags"

LaunchConfig :: struct {
	base_dir: string,
	game:     string,
}

LaunchOptions :: struct {
	basedir: string `usage:"Base directory containing game data directories."`,
	game:    string `usage:"Primary game directory to search before base."`,
	map_name: string `args:"name=map" usage:"Map name consumed by game code."`,
}

launch_config_parse :: proc(args: []string) -> LaunchConfig {
	options := launch_options_parse(args)
	config := LaunchConfig {
		base_dir = ".",
		game = options.game,
	}
	if options.basedir != "" {
		config.base_dir = options.basedir
	}
	return config
}

launch_options_parse :: proc(args: []string) -> LaunchOptions {
	runtime_args := make([dynamic]string, 0, len(args) + 1, context.temp_allocator)
	append(&runtime_args, "arctic-char")
	for arg in args {
		append(&runtime_args, arg)
	}

	options: LaunchOptions
	flags.parse_or_exit(&options, runtime_args[:], .Unix)
	return options
}

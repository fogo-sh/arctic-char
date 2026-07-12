package engine

import flags "core:flags"
import "core:strings"

LaunchConfig :: struct {
	base_dir: string,
	game:     string,
	fullscreen: bool,
	window_title: string,
	window_width: i32,
	window_height: i32,
	window_x: i32,
	window_y: i32,
	window_position_set: bool,
}

LaunchOptions :: struct {
	basedir: string `usage:"Base directory containing game data directories."`,
	game:    string `usage:"Primary game directory to search before base."`,
	map_name: string `args:"name=map" usage:"Map name consumed by game code."`,
	connect: string `usage:"Server address consumed by game code."`,
	port:    u16    `usage:"Server UDP port consumed by game code."`,
	content_id: u32 `usage:"Map/content identifier consumed by game code."`,
	fullscreen: bool `usage:"Start the window in fullscreen mode."`,
	window_title: string `usage:"Initial SDL window title."`,
	window_width:  i32    `usage:"Initial SDL window width."`,
	window_height: i32    `usage:"Initial SDL window height."`,
	window_x:      i32    `usage:"Initial SDL window x position."`,
	window_y:      i32    `usage:"Initial SDL window y position."`,
}

launch_config_parse :: proc(args: []string) -> LaunchConfig {
	options := launch_options_parse(args)
	config := LaunchConfig {
		game = options.game,
		fullscreen = options.fullscreen,
		window_title = options.window_title,
		window_width = options.window_width,
		window_height = options.window_height,
		window_x = options.window_x,
		window_y = options.window_y,
		window_position_set = launch_arg_present(args, "--window-x") || launch_arg_present(args, "--window-y"),
	}
	if options.basedir != "" {
		config.base_dir = options.basedir
	}
	return config
}

launch_arg_present :: proc(args: []string, name: string) -> bool {
	for arg in args {
		if arg == name || (len(arg) > len(name) && strings.has_prefix(arg, name) && arg[len(name)] == '=') {
			return true
		}
	}
	return false
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

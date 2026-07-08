package engine

import "core:os"
import "core:strings"

LaunchConfig :: struct {
	base_dir: string,
	game:     string,
	map_name: string,
}

launch_config_parse :: proc() -> LaunchConfig {
	config := LaunchConfig {
		base_dir = ".",
		map_name = "test",
	}

	args := os.args[1:]
	for i := 0; i < len(args); i += 1 {
		arg := args[i]
		switch arg {
		case "-basedir":
			if i + 1 < len(args) {
				i += 1
				config.base_dir = args[i]
			}
		case "-game":
			if i + 1 < len(args) {
				i += 1
				config.game = args[i]
			}
		case "+map":
			if i + 1 < len(args) {
				i += 1
				config.map_name = args[i]
			}
		}
	}
	return config
}

launch_config_map_qpath :: proc(config: LaunchConfig, allocator := context.allocator) -> string {
	// Make `+map test` resolve to `maps/test.map`
	qpath, err := strings.concatenate({"maps/", config.map_name, ".map"}, allocator)
	assert(err == nil)
	return qpath
}

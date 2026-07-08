package engine

import "core:os"

LaunchConfig :: struct {
	base_dir: string,
	game:     string,
}

launch_config_parse :: proc() -> LaunchConfig {
	config := LaunchConfig {
		base_dir = ".",
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
		}
	}
	return config
}

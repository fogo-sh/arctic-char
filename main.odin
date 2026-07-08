package main

import game "./src/game"

main :: proc() {
	game.Run()
}

when ODIN_OS == .Windows {
	@(export)
	NvOptimusEnablement: u32 = 1

	@(export)
	AmdPowerXpressRequestHighPerformance: u32 = 1
}

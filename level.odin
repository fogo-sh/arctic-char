package main

LevelAsset :: struct {
	source:       QuakeMap,
	render_mesh:  CpuMesh,
	player_spawn: MapPlayerSpawn,
}

level_load :: proc(fs: ^GameFS, qpath: string, allocator := context.allocator) -> LevelAsset {
	level: LevelAsset
	level.source = quake_map_load(fs, qpath, allocator)
	level.render_mesh = create_map_mesh(&level.source, allocator)
	level.player_spawn = PLAYER_DEFAULT_SPAWN
	if spawn, ok := quake_map_find_player_spawn(&level.source); ok {
		level.player_spawn = spawn
	}
	return level
}

level_destroy :: proc(level: ^LevelAsset) {
	cpu_mesh_destroy(&level.render_mesh)
	quake_map_destroy(&level.source)
	level^ = {}
}

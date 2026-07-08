package game

import "core:log"

LevelAsset :: struct {
	source:       QuakeMap,
	render_mesh:  CpuMesh,
	player_spawn: MapPlayerSpawn,
	diagnostics:  LevelDiagnostics,
}

LevelDiagnostics :: struct {
	entities:          int,
	worldspawn_brushes: int,
	worldspawn_faces:   int,
	has_player_spawn:   bool,
}

level_load :: proc(fs: ^GameFS, qpath: string, allocator := context.allocator) -> LevelAsset {
	level: LevelAsset
	level.source = quake_map_load(fs, qpath, allocator)
	level.diagnostics = level_collect_diagnostics(&level.source)
	assert(level.diagnostics.worldspawn_brushes > 0)
	assert(level.diagnostics.worldspawn_faces > 0)
	log.debugf(
		"Compiled level: entities=%d worldspawn_brushes=%d worldspawn_faces=%d has_player_spawn=%v",
		level.diagnostics.entities,
		level.diagnostics.worldspawn_brushes,
		level.diagnostics.worldspawn_faces,
		level.diagnostics.has_player_spawn,
	)

	level.render_mesh = create_map_mesh(&level.source, allocator)
	level.player_spawn = PLAYER_DEFAULT_SPAWN
	if spawn, ok := quake_map_find_player_spawn(&level.source); ok {
		level.player_spawn = spawn
	}
	return level
}

level_collect_diagnostics :: proc(qmap: ^QuakeMap) -> LevelDiagnostics {
	diagnostics := LevelDiagnostics{entities = len(qmap.entities)}
	for &entity in qmap.entities {
		classname, ok := map_entity_property(&entity, "classname")
		if !ok {
			continue
		}
		switch classname {
		case "worldspawn":
			diagnostics.worldspawn_brushes += len(entity.brushes)
			for &brush in entity.brushes {
				diagnostics.worldspawn_faces += len(brush.faces)
			}
		case "player":
			_, diagnostics.has_player_spawn = map_entity_property(&entity, "origin")
		}
	}
	return diagnostics
}

level_destroy :: proc(level: ^LevelAsset) {
	cpu_mesh_destroy(&level.render_mesh)
	quake_map_destroy(&level.source)
	level^ = {}
}

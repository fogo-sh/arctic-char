package main

import "core:log"
import "core:os"

SCENE_MESH_SUZANNE :: MeshHandle(0)
SCENE_MESH_MAP :: MeshHandle(1)

SceneAssets :: struct {
	suzanne_mesh:   CpuMesh,
	collision_mesh: CpuMesh,
	level:          LevelAsset,
}

scene_assets_load :: proc() -> SceneAssets {
	assets: SceneAssets
	assets.suzanne_mesh = load_glb_mesh("./assets/suzanne.glb")
	assets.level = level_load("./assets/test.map")

	assert(os.is_file("./assets/suzanne_collision.glb"))
	assets.collision_mesh = load_glb_mesh("./assets/suzanne_collision.glb")
	log.debugf(
		"Loaded Suzanne collision mesh: vertices=%d indices=%d",
		len(assets.collision_mesh.vertices),
		len(assets.collision_mesh.indices),
	)

	log.debugf("Loaded Suzanne: vertices=%d indices=%d", len(assets.suzanne_mesh.vertices), len(assets.suzanne_mesh.indices))
	return assets
}

scene_assets_destroy :: proc(assets: ^SceneAssets) {
	cpu_mesh_destroy(&assets.suzanne_mesh)
	cpu_mesh_destroy(&assets.collision_mesh)
	level_destroy(&assets.level)
	assets^ = {}
}

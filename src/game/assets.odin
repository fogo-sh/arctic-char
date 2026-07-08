package game

import "base:runtime"
import "core:log"

SceneAssets :: struct {
	suzanne_mesh:   CpuMesh,
	collision_mesh: CpuMesh,
	level:          LevelAsset,
	suzanne_handle: MeshHandle,
	map_handle:     MeshHandle,
	default_material: MaterialHandle,
}

scene_assets_load :: proc(fs: ^GameFS, config: LaunchConfig) -> SceneAssets {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	assets: SceneAssets
	assets.suzanne_mesh = load_glb_mesh(fs, "models/suzanne.glb")
	map_qpath := launch_config_map_qpath(config, context.temp_allocator)
	assets.level = level_load(fs, map_qpath)

	assets.collision_mesh = load_glb_mesh(fs, "models/suzanne_collision.glb")
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

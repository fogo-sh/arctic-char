package game

import "base:runtime"
import "core:log"
import engine "../engine"

LoadedSceneAssets :: struct {
	suzanne_mesh:   CpuMesh,
	collision_mesh: CpuMesh,
	level:          LevelAsset,
}

SceneGpuResources :: struct {
	suzanne_handle: MeshHandle,
	map_handle:     MeshHandle,
}

scene_assets_load :: proc(fs: ^GameFS, map_qpath: string) -> LoadedSceneAssets {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	assets: LoadedSceneAssets
	assets.suzanne_mesh = engine.load_glb_mesh(fs, "models/suzanne.glb")
	assets.level = level_load(fs, map_qpath)

	assets.collision_mesh = engine.load_glb_mesh(fs, "models/suzanne_collision.glb")
	log.debugf(
		"Loaded Suzanne collision mesh: vertices=%d indices=%d",
		len(assets.collision_mesh.vertices),
		len(assets.collision_mesh.indices),
	)

	log.debugf("Loaded Suzanne: vertices=%d indices=%d", len(assets.suzanne_mesh.vertices), len(assets.suzanne_mesh.indices))
	return assets
}

scene_gpu_resources_upload :: proc(renderer: ^Renderer, assets: ^LoadedSceneAssets) -> SceneGpuResources {
	upload := engine.renderer_begin_upload(renderer)
	gpu := SceneGpuResources{
		suzanne_handle = engine.renderer_upload_mesh(&upload, &assets.suzanne_mesh),
		map_handle = engine.renderer_upload_mesh(&upload, &assets.level.render_mesh),
	}
	engine.renderer_end_upload(&upload)
	return gpu
}

scene_assets_destroy :: proc(assets: ^LoadedSceneAssets) {
	engine.cpu_mesh_destroy(&assets.suzanne_mesh)
	engine.cpu_mesh_destroy(&assets.collision_mesh)
	level_destroy(&assets.level)
	assets^ = {}
}

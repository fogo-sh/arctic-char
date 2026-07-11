package game

import "base:runtime"
import "core:log"
import "core:strings"
import engine "../engine"

MAX_PROP_ASSETS :: 32
DEFAULT_PROP_MODEL :: "models/suzanne.glb"

LoadedPropAsset :: struct {
	model_path:           string,
	collision_model_path: string,
	render_mesh:          CpuMesh,
	collision_mesh:       CpuMesh,
}

LoadedSceneAssets :: struct {
	prop_assets: [dynamic]LoadedPropAsset,
	level:       LevelAsset,
}

SceneGpuResources :: struct {
	prop_handles: [MAX_PROP_ASSETS]MeshHandle,
	prop_count:   int,
	map_handle:   MeshHandle,
}

scene_assets_load :: proc(fs: ^GameFS, map_qpath: string) -> LoadedSceneAssets {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	assets: LoadedSceneAssets
	assets.level = level_load(fs, map_qpath)
	assets.prop_assets = make([dynamic]LoadedPropAsset, 0, 8)
	scene_assets_load_map_props(&assets, fs, &assets.level.source)
	return assets
}

scene_gpu_resources_upload :: proc(renderer: ^Renderer, assets: ^LoadedSceneAssets) -> SceneGpuResources {
	upload := engine.renderer_begin_upload(renderer)
	gpu := SceneGpuResources{prop_count = len(assets.prop_assets)}
	for &asset, i in assets.prop_assets {
		gpu.prop_handles[i] = engine.renderer_upload_mesh(&upload, &asset.render_mesh)
	}
	gpu.map_handle = engine.renderer_upload_mesh(&upload, &assets.level.render_mesh)
	engine.renderer_end_upload(&upload)
	return gpu
}

scene_assets_destroy :: proc(assets: ^LoadedSceneAssets) {
	for &asset in assets.prop_assets {
		engine.cpu_mesh_destroy(&asset.render_mesh)
		engine.cpu_mesh_destroy(&asset.collision_mesh)
		delete(asset.model_path)
		delete(asset.collision_model_path)
	}
	delete(assets.prop_assets)
	level_destroy(&assets.level)
	assets^ = {}
}

scene_assets_load_map_props :: proc(assets: ^LoadedSceneAssets, fs: ^GameFS, qmap: ^QuakeMap) {
	for &entity in qmap.entities {
		classname, has_classname := map_entity_property(&entity, "classname")
		if !has_classname || (classname != "prop_physics" && classname != "spawner_prop") {
			continue
		}
		model_path := scene_entity_prop_model(&entity)
		collision_path, has_collision_path := map_entity_property(&entity, "collision_model")
		if has_collision_path && collision_path != "" {
			scene_assets_ensure_prop(assets, fs, model_path, collision_path)
		} else {
			scene_assets_ensure_prop(assets, fs, model_path)
		}
	}
}

scene_assets_ensure_prop :: proc(assets: ^LoadedSceneAssets, fs: ^GameFS, model_path: string, collision_model_path := "") -> u16 {
	for &asset, i in assets.prop_assets {
		if asset.model_path == model_path {
			return u16(i)
		}
	}
	assert(len(assets.prop_assets) < MAX_PROP_ASSETS)

	owned_model_path := strings.clone(model_path)
	owned_collision_path := strings.clone(collision_model_path)
	if owned_collision_path == "" {
		owned_collision_path = prop_default_collision_model_path(model_path)
	}
	asset := LoadedPropAsset{
		model_path = owned_model_path,
		collision_model_path = owned_collision_path,
		render_mesh = engine.load_glb_mesh(fs, owned_model_path),
		collision_mesh = engine.load_glb_mesh(fs, owned_collision_path),
	}
	log.debugf("Loaded prop model=%s vertices=%d indices=%d collision=%s collision_vertices=%d", asset.model_path, len(asset.render_mesh.vertices), len(asset.render_mesh.indices), asset.collision_model_path, len(asset.collision_mesh.vertices))
	append(&assets.prop_assets, asset)
	return u16(len(assets.prop_assets) - 1)
}

prop_default_collision_model_path :: proc(model_path: string) -> string {
	dot := len(model_path)
	for i := len(model_path) - 1; i >= 0; i -= 1 {
		if model_path[i] == '.' {
			dot = i
			break
		}
		if model_path[i] == '/' {
			break
		}
	}
	path, err := strings.concatenate({model_path[:dot], "_collision", model_path[dot:]})
	assert(err == nil)
	return path
}

scene_assets_prop_index :: proc(assets: ^LoadedSceneAssets, model_path: string) -> u16 {
	for &asset, i in assets.prop_assets {
		if asset.model_path == model_path {
			return u16(i)
		}
	}
	return 0
}

scene_entity_prop_model :: proc(entity: ^MapEntity) -> string {
	model_path, ok := map_entity_property(entity, "model")
	if !ok || model_path == "" {
		return DEFAULT_PROP_MODEL
	}
	return model_path
}

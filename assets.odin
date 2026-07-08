package main

import "core:log"
import "core:os"

SCENE_MESH_SUZANNE :: MeshHandle(0)
SCENE_MESH_GROUND :: MeshHandle(1)

CollisionSource :: enum {
	Authored,
	Generated_From_Visual,
}

SceneAssets :: struct {
	render_meshes:     [2]CpuMesh,
	collision_mesh:    CpuMesh,
	collision_source:  CollisionSource,
}

scene_assets_load :: proc() -> SceneAssets {
	assets: SceneAssets
	assets.render_meshes[int(SCENE_MESH_SUZANNE)] = load_glb_mesh("./assets/suzanne.glb")
	assets.render_meshes[int(SCENE_MESH_GROUND)] = create_ground_mesh()

	if os.is_file("./assets/suzanne_collision.glb") {
		assets.collision_mesh = load_glb_mesh("./assets/suzanne_collision.glb")
		assets.collision_source = .Authored
		log.debugf(
			"Loaded Suzanne collision mesh: vertices=%d indices=%d",
			len(assets.collision_mesh.vertices),
			len(assets.collision_mesh.indices),
		)
	} else {
		assets.collision_source = .Generated_From_Visual
		log.warn("assets/suzanne_collision.glb missing; generating a runtime convex hull from visual Suzanne")
	}

	suzanne_mesh := &assets.render_meshes[int(SCENE_MESH_SUZANNE)]
	log.debugf("Loaded Suzanne: vertices=%d indices=%d", len(suzanne_mesh.vertices), len(suzanne_mesh.indices))
	return assets
}

scene_assets_destroy :: proc(assets: ^SceneAssets) {
	for &mesh in assets.render_meshes {
		cpu_mesh_destroy(&mesh)
	}
	if assets.collision_source == .Authored {
		cpu_mesh_destroy(&assets.collision_mesh)
	}
	assets^ = {}
}

scene_assets_collision_mesh :: proc(assets: ^SceneAssets) -> ^CpuMesh {
	if assets.collision_source == .Authored {
		return &assets.collision_mesh
	}
	return &assets.render_meshes[int(SCENE_MESH_SUZANNE)]
}

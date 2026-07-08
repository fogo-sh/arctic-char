package game

import engine "../engine"

Color :: engine.Color
CpuMesh :: engine.CpuMesh
GameFS :: engine.GameFS
LaunchConfig :: engine.LaunchConfig
MaterialHandle :: engine.MaterialHandle
MeshHandle :: engine.MeshHandle
PhysicsWorld :: engine.PhysicsWorld
PlayerLookInput :: engine.PlayerLookInput
PlayerMoveInput :: engine.PlayerMoveInput
RenderItem :: engine.RenderItem
RenderPassGlobals :: engine.RenderPassGlobals
Renderer :: engine.Renderer
Vec2 :: engine.Vec2
Vec3 :: engine.Vec3
VertexData :: engine.VertexData

PHYSICS_STEP_TIME : f32 : engine.PHYSICS_STEP_TIME

cpu_mesh_destroy :: engine.cpu_mesh_destroy
game_fs_read_file :: engine.game_fs_read_file
game_fs_modification_time :: engine.game_fs_modification_time
launch_config_map_qpath :: engine.launch_config_map_qpath
load_glb_mesh :: engine.load_glb_mesh
physics_create :: engine.physics_create
physics_create_suzanne_body :: engine.physics_create_suzanne_body
physics_body_matrix :: engine.physics_body_matrix
physics_destroy :: engine.physics_destroy
physics_player_query_filter :: engine.physics_player_query_filter
physics_replace_map_mesh :: engine.physics_replace_map_mesh
physics_step :: engine.physics_step
renderer_begin_upload :: engine.renderer_begin_upload
renderer_default_material :: engine.renderer_default_material
renderer_end_upload :: engine.renderer_end_upload
renderer_replace_mesh :: engine.renderer_replace_mesh
renderer_upload_mesh :: engine.renderer_upload_mesh

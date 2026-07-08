package game

import engine "../engine"

Color :: engine.Color
CpuMesh :: engine.CpuMesh
GameFS :: engine.GameFS
LaunchConfig :: engine.LaunchConfig
InputButton :: engine.InputButton
InputState :: engine.InputState
MaterialHandle :: engine.MaterialHandle
MeshHandle :: engine.MeshHandle
PhysicsWorld :: engine.PhysicsWorld
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
load_glb_mesh :: engine.load_glb_mesh
physics_create :: engine.physics_create
physics_body_matrix :: engine.physics_body_matrix
physics_create_static_mesh_data :: engine.physics_create_static_mesh_data
physics_destroy :: engine.physics_destroy
physics_step :: engine.physics_step
renderer_begin_upload :: engine.renderer_begin_upload
renderer_default_material :: engine.renderer_default_material
renderer_end_upload :: engine.renderer_end_upload
renderer_replace_mesh :: engine.renderer_replace_mesh
renderer_upload_mesh :: engine.renderer_upload_mesh

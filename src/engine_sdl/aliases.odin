package engine_sdl

import engine "../engine"

Color :: engine.Color
CpuMesh :: engine.CpuMesh
DebugLine :: engine.DebugLine
FrameTimingStats :: engine.FrameTimingStats
GameFS :: engine.GameFS
Game_API :: engine.Game_API
InputState :: engine.InputState
LaunchConfig :: engine.LaunchConfig
MeshHandle :: engine.MeshHandle
RenderEnvironment :: engine.RenderEnvironment
RendererApi :: engine.RendererApi
RendererStats :: engine.RendererStats
RenderFrame :: engine.RenderFrame
RenderItem :: engine.RenderItem
RenderPassGlobals :: engine.RenderPassGlobals
Text_Context :: engine.Text_Context
Text_Texture_Pack :: engine.Text_Texture_Pack
Text_Vertex :: engine.Text_Vertex
UiCommand :: engine.UiCommand
UiContext :: engine.UiContext
UiGeometryDraw :: engine.UiGeometryDraw
UiGeometryDrawKind :: engine.UiGeometryDrawKind
Vec3 :: engine.Vec3
VertexData :: engine.VertexData

TEXT_INDICES_PER_QUAD :: engine.TEXT_INDICES_PER_QUAD
TEXT_MAX_GLYPH_INDICES :: engine.TEXT_MAX_GLYPH_INDICES
TEXT_MAX_GLYPH_QUADS :: engine.TEXT_MAX_GLYPH_QUADS
TEXT_MAX_GLYPH_VERTICES :: engine.TEXT_MAX_GLYPH_VERTICES
TEXT_VERTICES_PER_QUAD :: engine.TEXT_VERTICES_PER_QUAD
UI_COMMAND_CAPACITY :: engine.UI_COMMAND_CAPACITY

game_fs_create :: engine.game_fs_create
game_fs_destroy :: engine.game_fs_destroy
input_begin_frame :: engine.input_begin_frame
input_set_key :: engine.input_set_key
performance_counter_now :: engine.performance_counter_now
performance_elapsed_ms :: engine.performance_elapsed_ms
text_begin :: engine.text_begin
text_context_destroy :: engine.text_context_destroy
text_draw :: engine.text_draw
text_font_load :: engine.text_font_load
text_font_load_ascii :: engine.text_font_load_ascii
text_font_process :: engine.text_font_process
text_pack_destroy :: engine.text_pack_destroy
text_register_font :: engine.text_register_font
text_vertex_count :: engine.text_vertex_count
ui_create :: engine.ui_create
ui_command_scaled :: engine.ui_command_scaled
ui_geometry_append_command :: engine.ui_geometry_append_command
ui_debug_hud_append_commands :: engine.ui_debug_hud_append_commands
ui_destroy :: engine.ui_destroy

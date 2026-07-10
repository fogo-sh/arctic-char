package engine

TextContext :: Context
TextTexturePack :: Texture_Pack_Result
TextVertex :: Vertex

TEXT_MAX_GLYPH_QUADS :: MAX_GLYPH_QUADS
TEXT_VERTICES_PER_QUAD :: VERTICES_PER_QUAD
TEXT_INDICES_PER_QUAD :: INDICES_PER_QUAD
TEXT_MAX_GLYPH_VERTICES :: MAX_GLYPH_VERTICES
TEXT_MAX_GLYPH_INDICES :: MAX_GLYPH_INDICES

text_begin :: begin
text_end :: end
text_vertex_count :: vertex_count
text_context_destroy :: destroy

text_font_load :: font_load
text_font_load_ascii :: font_load_ascii
text_register_font :: register_font
text_font_process :: font_process
text_pack_destroy :: pack_result_destroy
text_draw :: draw_text

package engine

import "core:log"
import "core:mem"
import sdl "vendor:sdl3"

TextVertexUniforms :: struct {
	mvp:      matrix[4, 4]f32,
	viewport: [2]f32,
}

TextFragmentUniforms :: struct {
	weight_boost: f32,
	padding:      [3]f32,
}

TextRenderer :: struct {
	ctx:             Text_Context,
	font_pack:       Text_Texture_Pack,
	pipeline:        ^sdl.GPUGraphicsPipeline,
	curve_texture:   ^sdl.GPUTexture,
	band_texture:    ^sdl.GPUTexture,
	vertex_buffer:   ^sdl.GPUBuffer,
	index_buffer:    ^sdl.GPUBuffer,
	transfer_buffer: ^sdl.GPUTransferBuffer,
	loaded:          bool,
}

renderer_create_text :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	sample_count: sdl.GPUSampleCount,
) -> ^TextRenderer {
	text := new(TextRenderer)
	assert(text != nil)
	text.pipeline = renderer_create_text_pipeline(gpu, window, sample_count)
	text.vertex_buffer = sdl.CreateGPUBuffer(gpu, {usage = {.VERTEX}, size = u32(TEXT_MAX_GLYPH_VERTICES * size_of(Text_Vertex))})
	text.index_buffer = sdl.CreateGPUBuffer(gpu, {usage = {.INDEX}, size = u32(TEXT_MAX_GLYPH_INDICES * size_of(u32))})
	text.transfer_buffer = sdl.CreateGPUTransferBuffer(gpu, {usage = .UPLOAD, size = u32(TEXT_MAX_GLYPH_VERTICES * size_of(Text_Vertex))})
	assert(text.pipeline != nil)
	assert(text.vertex_buffer != nil)
	assert(text.index_buffer != nil)
	assert(text.transfer_buffer != nil)

	font_path := "/System/Library/Fonts/Supplemental/Arial.ttf"
	font, font_ok := text_font_load(font_path)
	if !font_ok {
		log.warnf("Text font unavailable: %s", font_path)
		return text
	}
	text_font_load_ascii(&font)
	text_register_font(&text.ctx, font)
	text.font_pack = text_font_process(&text.ctx.font)
	text.curve_texture = renderer_create_text_texture(gpu, .R16G16B16A16_FLOAT, text.font_pack.curve_width, text.font_pack.curve_height)
	text.band_texture = renderer_create_text_texture(gpu, .R16G16_UINT, text.font_pack.band_width, text.font_pack.band_height)
	assert(text.curve_texture != nil)
	assert(text.band_texture != nil)

	renderer_upload_text_static(gpu, text)
	text.loaded = true
	return text
}

renderer_destroy_text :: proc(gpu: ^sdl.GPUDevice, text: ^TextRenderer) {
	if text == nil do return
	if text.transfer_buffer != nil do sdl.ReleaseGPUTransferBuffer(gpu, text.transfer_buffer)
	if text.index_buffer != nil do sdl.ReleaseGPUBuffer(gpu, text.index_buffer)
	if text.vertex_buffer != nil do sdl.ReleaseGPUBuffer(gpu, text.vertex_buffer)
	if text.band_texture != nil do sdl.ReleaseGPUTexture(gpu, text.band_texture)
	if text.curve_texture != nil do sdl.ReleaseGPUTexture(gpu, text.curve_texture)
	if text.pipeline != nil do sdl.ReleaseGPUGraphicsPipeline(gpu, text.pipeline)
	text_pack_destroy(&text.font_pack)
	text_context_destroy(&text.ctx)
	free(text)
}

renderer_create_text_pipeline :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	sample_count: sdl.GPUSampleCount,
) -> ^sdl.GPUGraphicsPipeline {
	vert_shader := load_shader(gpu, text_vert_shader_code, .VERTEX, 1, "TextVertexMain")
	frag_shader := load_shader(gpu, text_frag_shader_code, .FRAGMENT, 1, "TextFragmentMain", num_storage_textures = 2)
	assert(vert_shader != nil)
	assert(frag_shader != nil)
	defer sdl.ReleaseGPUShader(gpu, vert_shader)
	defer sdl.ReleaseGPUShader(gpu, frag_shader)

	vertex_buffer_description := sdl.GPUVertexBufferDescription {
		slot = 0,
		pitch = size_of(Text_Vertex),
		input_rate = .VERTEX,
	}
	vertex_attributes := [5]sdl.GPUVertexAttribute {
		{location = 0, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(Text_Vertex, pos))},
		{location = 1, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(Text_Vertex, tex))},
		{location = 2, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(Text_Vertex, jac))},
		{location = 3, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(Text_Vertex, bnd))},
		{location = 4, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(Text_Vertex, col))},
	}
	color_target_description := sdl.GPUColorTargetDescription {
		format = sdl.GetGPUSwapchainTextureFormat(gpu, window),
		blend_state = {
			src_color_blendfactor = .SRC_ALPHA,
			dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA,
			color_blend_op = .ADD,
			src_alpha_blendfactor = .ONE,
			dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA,
			alpha_blend_op = .ADD,
			enable_blend = true,
		},
	}
	target_info := sdl.GPUGraphicsPipelineTargetInfo {
		num_color_targets = 1,
		color_target_descriptions = &color_target_description,
		depth_stencil_format = .D32_FLOAT,
		has_depth_stencil_target = true,
	}

	pipeline := sdl.CreateGPUGraphicsPipeline(gpu, {
		vertex_shader = vert_shader,
		fragment_shader = frag_shader,
		vertex_input_state = {
			num_vertex_buffers = 1,
			vertex_buffer_descriptions = &vertex_buffer_description,
			num_vertex_attributes = len(vertex_attributes),
			vertex_attributes = &vertex_attributes[0],
		},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {cull_mode = .NONE, enable_depth_clip = true},
		multisample_state = {sample_count = sample_count},
		depth_stencil_state = {enable_depth_test = false, enable_depth_write = false},
		target_info = target_info,
	})
	assert(pipeline != nil)
	return pipeline
}

renderer_create_text_texture :: proc(gpu: ^sdl.GPUDevice, format: sdl.GPUTextureFormat, width, height: u32) -> ^sdl.GPUTexture {
	return sdl.CreateGPUTexture(gpu, {
		type = .D2,
		format = format,
		usage = {.GRAPHICS_STORAGE_READ},
		width = width,
		height = height,
		layer_count_or_depth = 1,
		num_levels = 1,
	})
}

renderer_upload_text_static :: proc(gpu: ^sdl.GPUDevice, text: ^TextRenderer) {
	index_size := TEXT_MAX_GLYPH_INDICES * size_of(u32)
	curve_size := len(text.font_pack.curve_data) * size_of([4]u16)
	band_size := len(text.font_pack.band_data) * size_of([2]u16)
	total_size := index_size + curve_size + band_size

	transfer := sdl.CreateGPUTransferBuffer(gpu, {usage = .UPLOAD, size = u32(total_size)})
	assert(transfer != nil)
	defer sdl.ReleaseGPUTransferBuffer(gpu, transfer)

	data := transmute([^]byte)sdl.MapGPUTransferBuffer(gpu, transfer, false)
	indices := transmute([^]u32)raw_data(data[:index_size])
	for quad in 0..<TEXT_MAX_GLYPH_QUADS {
		base_vertex := u32(quad * TEXT_VERTICES_PER_QUAD)
		base_index := quad * TEXT_INDICES_PER_QUAD
		indices[base_index + 0] = base_vertex + 0
		indices[base_index + 1] = base_vertex + 1
		indices[base_index + 2] = base_vertex + 2
		indices[base_index + 3] = base_vertex + 2
		indices[base_index + 4] = base_vertex + 3
		indices[base_index + 5] = base_vertex + 0
	}
	mem.copy(data[index_size:], raw_data(text.font_pack.curve_data), curve_size)
	mem.copy(data[index_size + curve_size:], raw_data(text.font_pack.band_data), band_size)
	sdl.UnmapGPUTransferBuffer(gpu, transfer)

	cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)
	assert(cmd_buf != nil)
	copy_pass := sdl.BeginGPUCopyPass(cmd_buf)
	assert(copy_pass != nil)
	sdl.UploadToGPUBuffer(copy_pass, {transfer_buffer = transfer, offset = 0}, {buffer = text.index_buffer, offset = 0, size = u32(index_size)}, false)
	sdl.UploadToGPUTexture(copy_pass, {transfer_buffer = transfer, offset = u32(index_size), pixels_per_row = text.font_pack.curve_width, rows_per_layer = text.font_pack.curve_height}, {texture = text.curve_texture, w = text.font_pack.curve_width, h = text.font_pack.curve_height, d = 1}, false)
	sdl.UploadToGPUTexture(copy_pass, {transfer_buffer = transfer, offset = u32(index_size + curve_size), pixels_per_row = text.font_pack.band_width, rows_per_layer = text.font_pack.band_height}, {texture = text.band_texture, w = text.font_pack.band_width, h = text.font_pack.band_height, d = 1}, false)
	sdl.EndGPUCopyPass(copy_pass)
	ok := sdl.SubmitGPUCommandBuffer(cmd_buf)
	assert(ok)
}

renderer_prepare_text :: proc(renderer: ^Renderer, cmd_buf: ^sdl.GPUCommandBuffer, commands: []UiCommand) {
	if renderer.text == nil || !renderer.text.loaded do return
	text := renderer.text
	text_begin(&text.ctx)
	font := &text.ctx.font
	for command in commands {
		if command.kind != .Text || command.text == "" do continue
		baseline_y := command.bounds.y + font.ascent * command.font_size
		text_draw(&text.ctx, command.text, command.bounds.x, baseline_y, command.font_size, command.color)
	}
	vertex_count := text_vertex_count(&text.ctx)
	if vertex_count == 0 do return

	vertex_data_size := int(vertex_count) * size_of(Text_Vertex)
	transfer_ptr := transmute([^]byte)sdl.MapGPUTransferBuffer(renderer.gpu, text.transfer_buffer, true)
	mem.copy(transfer_ptr[:], raw_data(text.ctx.vertices[:vertex_count]), vertex_data_size)
	sdl.UnmapGPUTransferBuffer(renderer.gpu, text.transfer_buffer)

	copy_pass := sdl.BeginGPUCopyPass(cmd_buf)
	assert(copy_pass != nil)
	sdl.UploadToGPUBuffer(copy_pass, {transfer_buffer = text.transfer_buffer, offset = 0}, {buffer = text.vertex_buffer, offset = 0, size = u32(vertex_data_size)}, true)
	sdl.EndGPUCopyPass(copy_pass)
}

renderer_draw_text :: proc(renderer: ^Renderer, cmd_buf: ^sdl.GPUCommandBuffer, render_pass: ^sdl.GPURenderPass, viewport: [2]i32) {
	if renderer.text == nil || !renderer.text.loaded || renderer.text.ctx.quad_count == 0 do return
	text := renderer.text

	sdl.BindGPUGraphicsPipeline(render_pass, text.pipeline)
	vertex_binding := sdl.GPUBufferBinding{buffer = text.vertex_buffer, offset = 0}
	index_binding := sdl.GPUBufferBinding{buffer = text.index_buffer, offset = 0}
	sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_binding, 1)
	sdl.BindGPUIndexBuffer(render_pass, index_binding, ._32BIT)
	storage_textures := [2]^sdl.GPUTexture{text.curve_texture, text.band_texture}
	sdl.BindGPUFragmentStorageTextures(render_pass, 0, &storage_textures[0], 2)

	vertex_uniforms := TextVertexUniforms{
		mvp = renderer_text_ortho(f32(viewport.x), f32(viewport.y)),
		viewport = {f32(viewport.x), f32(viewport.y)},
	}
	fragment_uniforms := TextFragmentUniforms{}
	sdl.PushGPUVertexUniformData(cmd_buf, 0, &vertex_uniforms, size_of(vertex_uniforms))
	sdl.PushGPUFragmentUniformData(cmd_buf, 0, &fragment_uniforms, size_of(fragment_uniforms))
	sdl.DrawGPUIndexedPrimitives(render_pass, text.ctx.quad_count * TEXT_INDICES_PER_QUAD, 1, 0, 0, 0)
	renderer.stats.draw_count += 1
	renderer.stats.triangle_count += int(text.ctx.quad_count) * 2
}

renderer_text_ortho :: proc(width, height: f32) -> matrix[4, 4]f32 {
	m: matrix[4, 4]f32
	m[0, 0] = 2.0 / width
	m[1, 1] = -2.0 / height
	m[2, 2] = 1.0
	m[3, 0] = -1.0
	m[3, 1] = 1.0
	m[3, 3] = 1.0
	return m
}

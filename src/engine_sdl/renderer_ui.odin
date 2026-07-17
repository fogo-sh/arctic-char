package engine_sdl

import "core:math"
import "core:mem"
import engine "../engine"
import sdl "vendor:sdl3"

UiVertexUniforms :: struct {
	rect:          [4]f32,
	color:         [4]f32,
	corner_radii:  [4]f32,
	border_widths: [4]f32,
	viewport:      [2]f32,
	padding:       [2]f32,
}

renderer_create_ui_pipeline :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	sample_count: sdl.GPUSampleCount,
) -> ^sdl.GPUGraphicsPipeline {
	vert_shader := load_shader(
		gpu,
		vert_shader_code,
		.VERTEX,
		num_uniform_buffers = 1,
		entrypoint_name = "VertexMain",
	)
	frag_shader := load_shader(
		gpu,
		frag_shader_code,
		.FRAGMENT,
		num_uniform_buffers = 1,
		entrypoint_name = "FragmentMain",
	)
	assert(vert_shader != nil)
	assert(frag_shader != nil)
	defer sdl.ReleaseGPUShader(gpu, vert_shader)
	defer sdl.ReleaseGPUShader(gpu, frag_shader)

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
	vertex_buffer_description := sdl.GPUVertexBufferDescription{slot = 0, pitch = size_of(VertexData), input_rate = .VERTEX}
	vertex_attributes := [4]sdl.GPUVertexAttribute{
		{location = 0, buffer_slot = 0, format = .FLOAT3, offset = 0},
		{location = 1, buffer_slot = 0, format = .FLOAT3, offset = u32(offset_of(VertexData, normal))},
		{location = 2, buffer_slot = 0, format = .FLOAT2, offset = u32(offset_of(VertexData, uv))},
		{location = 3, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(VertexData, color))},
	}
	target_info := sdl.GPUGraphicsPipelineTargetInfo {
		num_color_targets         = 1,
		color_target_descriptions = &color_target_description,
		depth_stencil_format      = .D32_FLOAT,
		has_depth_stencil_target  = true,
	}

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		{
			vertex_shader = vert_shader,
			fragment_shader = frag_shader,
			vertex_input_state = {
				num_vertex_buffers = 1,
				vertex_buffer_descriptions = &vertex_buffer_description,
				num_vertex_attributes = len(vertex_attributes),
				vertex_attributes = &vertex_attributes[0],
			},
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {
				cull_mode = .NONE,
				enable_depth_clip = true,
			},
			multisample_state = {
				sample_count = sample_count,
			},
			depth_stencil_state = {
				enable_depth_test = false,
				enable_depth_write = false,
			},
			target_info = target_info,
		},
	)
	assert(pipeline != nil)
	return pipeline
}

renderer_prepare_ui :: proc(renderer: ^Renderer, cmd_buf: ^sdl.GPUCommandBuffer, commands: []UiCommand, ui_scale: [2]f32) {
	clear(&renderer.ui_vertices)
	clear(&renderer.ui_draws)
	if len(commands) == 0 do return
	for command in commands {
		scaled := ui_command_scaled(command, ui_scale)
		#partial switch command.kind {
		case .Rectangle:
			ui_geometry_append_command(&renderer.ui_vertices, &renderer.ui_draws, scaled)
		case .Border:
			ui_geometry_append_command(&renderer.ui_vertices, &renderer.ui_draws, scaled)
		case .Text:
		case .ScissorStart:
			append(&renderer.ui_draws, UiGeometryDraw{kind = .ScissorStart, bounds = scaled.bounds})
		case .ScissorEnd:
			append(&renderer.ui_draws, UiGeometryDraw{kind = .ScissorEnd})
		}
	}
	if len(renderer.ui_vertices) == 0 do return
	data_size := len(renderer.ui_vertices) * size_of(VertexData)
	transfer_ptr := transmute([^]byte)sdl.MapGPUTransferBuffer(renderer.gpu, renderer.ui_transfer_buffer, true)
	mem.copy(transfer_ptr[:], raw_data(renderer.ui_vertices[:]), data_size)
	sdl.UnmapGPUTransferBuffer(renderer.gpu, renderer.ui_transfer_buffer)
	copy_pass := sdl.BeginGPUCopyPass(cmd_buf)
	assert(copy_pass != nil)
	sdl.UploadToGPUBuffer(copy_pass, {transfer_buffer = renderer.ui_transfer_buffer, offset = 0}, {buffer = renderer.ui_vertex_buffer, offset = 0, size = u32(data_size)}, true)
	sdl.EndGPUCopyPass(copy_pass)
}

renderer_draw_ui :: proc(
	renderer: ^Renderer,
	cmd_buf: ^sdl.GPUCommandBuffer,
	render_pass: ^sdl.GPURenderPass,
	commands: []UiCommand,
	viewport: [2]i32,
) {
	if len(renderer.ui_vertices) == 0 do return

	full_scissor := sdl.Rect{x = 0, y = 0, w = viewport.x, h = viewport.y}
	sdl.SetGPUScissor(render_pass, full_scissor)
	sdl.BindGPUGraphicsPipeline(render_pass, renderer.ui_pipeline)
	fragment_uniforms := FragmentUniforms{fog_distances = {1.0e9, 2.0e9, 0, 0}}
	sdl.PushGPUFragmentUniformData(cmd_buf, 0, &fragment_uniforms, size_of(fragment_uniforms))
	vertex_uniforms := WorldVertexUniforms{mvp = renderer_ui_ortho(f32(viewport.x), f32(viewport.y)), model_view = renderer_matrix4_identity()}
	sdl.PushGPUVertexUniformData(cmd_buf, 0, &vertex_uniforms, size_of(vertex_uniforms))
	vertex_buffer_binding := sdl.GPUBufferBinding{buffer = renderer.ui_vertex_buffer, offset = 0}
	sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)

	for draw in renderer.ui_draws {
		switch draw.kind {
		case .Geometry:
			if draw.count == 0 do continue
			sdl.DrawGPUPrimitives(render_pass, u32(draw.count), 1, u32(draw.start), 0)
			renderer.stats.draw_count += 1
			renderer.stats.triangle_count += draw.count / 3
		case .ScissorStart:
			sdl.SetGPUScissor(render_pass, renderer_ui_scissor(draw.bounds, viewport))
		case .ScissorEnd:
			sdl.SetGPUScissor(render_pass, full_scissor)
		}
	}

	sdl.SetGPUScissor(render_pass, full_scissor)
}

renderer_ui_scissor :: proc(bounds: [4]f32, viewport: [2]i32) -> sdl.Rect {
	x0 := max(i32(math.floor(bounds.x)), 0)
	y0 := max(i32(math.floor(bounds.y)), 0)
	x1 := min(i32(math.ceil(bounds.x + bounds.z)), viewport.x)
	y1 := min(i32(math.ceil(bounds.y + bounds.w)), viewport.y)
	return {x = x0, y = y0, w = max(x1 - x0, 0), h = max(y1 - y0, 0)}
}

renderer_ui_ortho :: proc(width, height: f32) -> matrix[4, 4]f32 {
	m: matrix[4, 4]f32
	m[0, 0] = 2.0 / width
	m[1, 1] = -2.0 / height
	m[2, 2] = 1.0
	m[3, 3] = 1.0
	m[0, 3] = -1.0
	m[1, 3] = 1.0
	return m
}

renderer_matrix4_identity :: proc() -> matrix[4, 4]f32 {
	m: matrix[4, 4]f32
	m[0, 0] = 1
	m[1, 1] = 1
	m[2, 2] = 1
	m[3, 3] = 1
	return m
}

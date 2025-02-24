package main

import clay "clay-odin"
import "core:log"
import "core:math"
import "core:mem"
import "core:strings"
import sdl "vendor:sdl3"

clayColorToSdlFColor :: proc(color: clay.Color) -> sdl.FColor {
	return sdl.FColor{color[0], color[1], color[2], color[3]}
}

measureText :: proc "c" (
	text: clay.StringSlice,
	config: ^clay.TextElementConfig,
	userData: rawptr,
) -> clay.Dimensions {
	width := cast(f32)(config.fontSize) * 0.6 * cast(f32)(text.length)
	height := cast(f32)(config.fontSize)
	return clay.Dimensions{width, height}
}

UIVertex :: struct {
	pos:   [3]f32,
	color: sdl.FColor,
	uv:    [2]f32,
}

make_quad_for_rectangle :: proc(
	bbox: clay.BoundingBox,
	color: clay.Color,
) -> (
	[4]UIVertex,
	[6]u16,
) {
	col := clayColorToSdlFColor(color)
	quad: [4]UIVertex = {
		{pos = [3]f32{bbox.x, bbox.y, 0}, color = col, uv = [2]f32{0, 0}},
		{pos = [3]f32{bbox.x + bbox.width, bbox.y, 0}, color = col, uv = [2]f32{1, 0}},
		{
			pos = [3]f32{bbox.x + bbox.width, bbox.y + bbox.height, 0},
			color = col,
			uv = [2]f32{1, 1},
		},
		{pos = [3]f32{bbox.x, bbox.y + bbox.height, 0}, color = col, uv = [2]f32{0, 1}},
	}
	idx: [6]u16 = {0, 1, 2, 2, 3, 0}
	return quad, idx
}

MAX_UI_RECTS :: 16
MAX_UI_VERTICES :: MAX_UI_RECTS * 4
MAX_UI_INDICES :: MAX_UI_RECTS * 6

ui_vertex_buffer: ^sdl.GPUBuffer = nil
ui_index_buffer: ^sdl.GPUBuffer = nil
ui_transfer_buffer: ^sdl.GPUTransferBuffer = nil

white_texture: ^sdl.GPUTexture = nil
sampler: ^sdl.GPUSampler = nil

claySdlGpuRenderMeasureText :: proc "c" (
	text: clay.StringSlice,
	config: ^clay.TextElementConfig,
	userData: rawptr,
) -> clay.Dimensions {
	textSize: clay.Dimensions = {0, 0}

	// TODO
	textSize = clay.Dimensions{100, 100}

	return textSize
}

claySdlGpuRenderInitialize :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
) -> ^sdl.GPUGraphicsPipeline {
	ui_vertex_buffer = sdl.CreateGPUBuffer(
		gpu,
		sdl.GPUBufferCreateInfo {
			usage = sdl.GPUBufferUsageFlags{.VERTEX},
			size = MAX_UI_VERTICES * size_of(UIVertex),
			props = 0,
		},
	)
	assert(ui_vertex_buffer != nil)

	ui_index_buffer = sdl.CreateGPUBuffer(
		gpu,
		sdl.GPUBufferCreateInfo {
			usage = sdl.GPUBufferUsageFlags{.INDEX},
			size = MAX_UI_INDICES * size_of(u16),
			props = 0,
		},
	)
	assert(ui_index_buffer != nil)

	ui_transfer_buffer = sdl.CreateGPUTransferBuffer(
		gpu,
		sdl.GPUTransferBufferCreateInfo {
			usage = sdl.GPUTransferBufferUsage.UPLOAD,
			size = MAX_UI_VERTICES * size_of(UIVertex) + MAX_UI_INDICES * size_of(u16),
			props = 0,
		},
	)
	assert(ui_transfer_buffer != nil)

	vert_shader := load_shader(
		gpu,
		ui_vert_shader_code,
		.VERTEX,
		num_uniform_buffers = 1,
		num_samplers = 0,
	)
	frag_shader := load_shader(
		gpu,
		ui_frag_shader_code,
		.FRAGMENT,
		num_uniform_buffers = 0,
		num_samplers = 1,
	)

	mesh_vertex_buffer_description := sdl.GPUVertexBufferDescription {
		slot               = 0,
		pitch              = size_of(f32) * (3 + 4 + 2), // float3 pos + float4 color + float2 uv
		input_rate         = .VERTEX,
		instance_step_rate = 0,
	}
	mesh_vertex_attribute_position := sdl.GPUVertexAttribute {
		location    = 0,
		buffer_slot = 0,
		format      = .FLOAT3,
		offset      = 0,
	}
	mesh_vertex_attribute_color := sdl.GPUVertexAttribute {
		location    = 1,
		buffer_slot = 0,
		format      = .FLOAT4,
		offset      = size_of(f32) * 3, // After float3 position
	}
	mesh_vertex_attribute_uv := sdl.GPUVertexAttribute {
		location    = 2,
		buffer_slot = 0,
		format      = .FLOAT2,
		offset      = size_of(f32) * (3 + 4), // After float3 position + float4 color
	}
	mesh_vertex_attributes := [3]sdl.GPUVertexAttribute {
		mesh_vertex_attribute_position,
		mesh_vertex_attribute_color,
		mesh_vertex_attribute_uv,
	}

	target_info: sdl.GPUGraphicsPipelineTargetInfo = sdl.GPUGraphicsPipelineTargetInfo {
		num_color_targets         = 1,
		color_target_descriptions = &sdl.GPUColorTargetDescription {
			format = sdl.GetGPUSwapchainTextureFormat(gpu, window),
			blend_state = sdl.GPUColorTargetBlendState{},
		},
	}

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		{
			vertex_shader = vert_shader,
			fragment_shader = frag_shader,
			vertex_input_state = {
				num_vertex_buffers = 1,
				vertex_buffer_descriptions = &mesh_vertex_buffer_description,
				num_vertex_attributes = 3,
				vertex_attributes = &mesh_vertex_attributes[0],
			},
			primitive_type = .TRIANGLELIST,
			rasterizer_state = sdl.GPURasterizerState{},
			depth_stencil_state = sdl.GPUDepthStencilState{},
			target_info = target_info,
		},
	)
	assert(pipeline != nil)

	white_texture = sdl.CreateGPUTexture(
		gpu,
		{
			format = .R8G8B8A8_UNORM,
			usage = {.SAMPLER},
			width = 1,
			height = 1,
			layer_count_or_depth = 1,
			num_levels = 1,
		},
	)
	assert(white_texture != nil)

	sampler = sdl.CreateGPUSampler(gpu, {})
	assert(sampler != nil)

	return pipeline
}

claySdlGpuRender :: proc(
	renderCommands: ^clay.ClayArray(clay.RenderCommand),
	gpu: ^sdl.GPUDevice,
	cmd_buf: ^sdl.GPUCommandBuffer,
	render_pass: ^sdl.GPURenderPass,
	ui_projection: matrix[4, 4]f32,
) {
	batch_vertices: []UIVertex = make([]UIVertex, MAX_UI_RECTS * 4)
	defer delete(batch_vertices)
	batch_indices: []u16 = make([]u16, MAX_UI_RECTS * 6)
	defer delete(batch_indices)
	vertex_count: int = 0
	index_count: int = 0

	for i in 0 ..< int(renderCommands.length) {
		cmd := clay.RenderCommandArray_Get(renderCommands, cast(i32)i)

		#partial switch (cmd.commandType) {
		case clay.RenderCommandType.Rectangle:
			{
				rect_config := cmd.renderData.rectangle
				quad, indices := make_quad_for_rectangle(
					cmd.boundingBox,
					rect_config.backgroundColor,
				)

				vertex_offset := cast(u16)(vertex_count)
				mem.copy(
					raw_data(batch_vertices[vertex_count:vertex_count + 4]),
					raw_data(quad[:]),
					4 * size_of(UIVertex),
				)
				vertex_count += 4

				for j := 0; j < 6; j += 1 {
					batch_indices[index_count + j] = indices[j] + vertex_offset
				}
				index_count += 6
			}
			default: {}
		}
	}

	if vertex_count == 0 {
		return
	}

	white_pixel: [4]u8 = {255, 255, 255, 255}

	vtx_data_size := cast(u32)(vertex_count) * size_of(UIVertex)
	idx_data_size := cast(u32)(index_count) * size_of(u16)
	total_size := vtx_data_size + idx_data_size + size_of(white_pixel)

	transfer_buffer_ptr := transmute([^]byte)sdl.MapGPUTransferBuffer(
		gpu,
		ui_transfer_buffer,
		false,
	)
	assert(transfer_buffer_ptr != nil)
	transfer_buffer_offset := 0

	mem.copy(
		transfer_buffer_ptr[transfer_buffer_offset:],
		raw_data(batch_vertices[:vertex_count]),
		int(vtx_data_size),
	)

	transfer_buffer_offset += int(vtx_data_size)
	mem.copy(
		transfer_buffer_ptr[transfer_buffer_offset:],
		raw_data(batch_indices[:index_count]),
		int(idx_data_size),
	)
	transfer_buffer_offset += int(idx_data_size)

	mem.copy(
		transfer_buffer_ptr[transfer_buffer_offset:],
		raw_data(&white_pixel),
		size_of(white_pixel),
	)

	transfer_buffer_offset += size_of(white_pixel)

	sdl.UnmapGPUTransferBuffer(gpu, ui_transfer_buffer)

	copy_cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_cmd_buf)

	buffer_offset: u32 = 0

	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = ui_transfer_buffer, offset = buffer_offset},
		{buffer = ui_vertex_buffer, offset = 0, size = vtx_data_size},
		false,
	)

	buffer_offset += vtx_data_size

	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = ui_transfer_buffer, offset = buffer_offset},
		{buffer = ui_index_buffer, offset = 0, size = idx_data_size},
		false,
	)

	buffer_offset += idx_data_size

	sdl.UploadToGPUTexture(
		copy_pass,
		{transfer_buffer = ui_transfer_buffer, offset = buffer_offset},
		{texture = white_texture, w = 1, h = 1, d = 1},
		false,
	)

	sdl.EndGPUCopyPass(copy_pass)
	ok := sdl.SubmitGPUCommandBuffer(copy_cmd_buf)
	assert(ok)

	vb_binding := sdl.GPUBufferBinding {
		buffer = ui_vertex_buffer,
		offset = 0,
	}
	ib_binding := sdl.GPUBufferBinding {
		buffer = ui_index_buffer,
		offset = 0,
	}
	sdl.BindGPUVertexBuffers(render_pass, 0, &vb_binding, 1)
	sdl.BindGPUIndexBuffer(render_pass, ib_binding, sdl.GPUIndexElementSize._16BIT)

	ui_proj_copy := ui_projection
	ui_projection_ptr := raw_data(&ui_proj_copy)
	sdl.PushGPUVertexUniformData(cmd_buf, 0, ui_projection_ptr, size_of(ui_projection))

	sdl.BindGPUFragmentSamplers(
		render_pass,
		0,
		&(sdl.GPUTextureSamplerBinding{texture = white_texture, sampler = sampler}),
		1,
	)

	sdl.DrawGPUIndexedPrimitives(render_pass, cast(u32)(index_count), 1, 0, 0, 0)
}

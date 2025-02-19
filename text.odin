package main

import "core:mem"
import sdl "vendor:sdl3"
import "vendor:stb/easy_font"

TextUBO :: struct {
	ortho: matrix[4, 4]f32,
}

TextDrawCommand :: struct {
	offset: u32,
	count:  u32,
	color:  [4]f32,
}

accumulate_text :: proc(
	text: string,
	x, y: f32,
	color: [4]f32,
	scale: f32,
	vertices: ^[4096]f32,
	vertex_offset: ^u32,
	draw_commands: ^[16]TextDrawCommand,
	draw_count: ^u32,
) {
	quads: [200]easy_font.Quad
	num_quads := easy_font.print(x, y, text, 0, quads[:], scale)
	base_offset := vertex_offset^

	for i in 0 ..< num_quads {
		q := quads[i]

		(vertices^)[base_offset + 0] = q.tl.v.x
		(vertices^)[base_offset + 1] = q.tl.v.y
		(vertices^)[base_offset + 2] = 0
		(vertices^)[base_offset + 3] = 1

		(vertices^)[base_offset + 4] = q.tr.v.x
		(vertices^)[base_offset + 5] = q.tr.v.y
		(vertices^)[base_offset + 6] = 0
		(vertices^)[base_offset + 7] = 1

		(vertices^)[base_offset + 8] = q.br.v.x
		(vertices^)[base_offset + 9] = q.br.v.y
		(vertices^)[base_offset + 10] = 0
		(vertices^)[base_offset + 11] = 1

		(vertices^)[base_offset + 12] = q.tl.v.x
		(vertices^)[base_offset + 13] = q.tl.v.y
		(vertices^)[base_offset + 14] = 0
		(vertices^)[base_offset + 15] = 1

		(vertices^)[base_offset + 16] = q.br.v.x
		(vertices^)[base_offset + 17] = q.br.v.y
		(vertices^)[base_offset + 18] = 0
		(vertices^)[base_offset + 19] = 1

		(vertices^)[base_offset + 20] = q.bl.v.x
		(vertices^)[base_offset + 21] = q.bl.v.y
		(vertices^)[base_offset + 22] = 0
		(vertices^)[base_offset + 23] = 1

		base_offset += 24
	}

	draw_commands[draw_count^] = TextDrawCommand {
		offset = vertex_offset^,
		count  = u32(num_quads * 6),
		color  = color,
	}
	draw_count^ += 1
	vertex_offset^ = base_offset
}

setup_text_pipeline :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	vertex_buffer_description: ^sdl.GPUVertexBufferDescription,
	vertex_attribute_description: ^sdl.GPUVertexAttribute,
) -> (
	pipeline: ^sdl.GPUGraphicsPipeline,
	vertex_buffer: ^sdl.GPUBuffer,
) {
	text_vert_shader := load_shader(gpu, text_vert_shader_code, .VERTEX, 1)
	text_frag_shader := load_shader(gpu, text_frag_shader_code, .FRAGMENT, 1)

	text_pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		{
			vertex_shader = text_vert_shader,
			fragment_shader = text_frag_shader,
			vertex_input_state = {
				num_vertex_buffers = 1,
				vertex_buffer_descriptions = vertex_buffer_description,
				num_vertex_attributes = 1,
				vertex_attributes = vertex_attribute_description,
			},
			primitive_type = .TRIANGLELIST,
			target_info = {
				num_color_targets = 1,
				color_target_descriptions = &(sdl.GPUColorTargetDescription {
						format = sdl.GetGPUSwapchainTextureFormat(gpu, window),
					}),
			},
		},
	)

	sdl.ReleaseGPUShader(gpu, text_vert_shader)
	sdl.ReleaseGPUShader(gpu, text_frag_shader)

	text_gpu_vertex_buffer := sdl.CreateGPUBuffer(gpu, {usage = {.VERTEX}, size = 4096, props = 0})
	text_transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = 4096, props = 0},
	)

	return text_pipeline, text_gpu_vertex_buffer
}

render_text :: proc(
	gpu: ^sdl.GPUDevice,
	cmd_buf: ^sdl.GPUCommandBuffer,
	render_pass: ^sdl.GPURenderPass,
	text_pipeline: ^sdl.GPUGraphicsPipeline,
	text_gpu_vertex_buffer: ^sdl.GPUBuffer,
	win_size: [2]i32,
	text_vertices: ^[4096]f32,
	vertex_offset: u32,
	draw_commands: ^[16]TextDrawCommand,
	draw_count: u32,
) {
	ortho_mat := orthographic_matrix(0, f32(win_size.x), f32(win_size.y), 0, -1, 1)
	text_ubo := TextUBO {
		ortho = ortho_mat,
	}
	sdl.PushGPUVertexUniformData(cmd_buf, 0, &text_ubo, size_of(text_ubo))

	size_to_copy := vertex_offset * size_of(f32)
	transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = size_to_copy, props = 0},
	)
	tb_ptr := sdl.MapGPUTransferBuffer(gpu, transfer_buffer, false)
	mem.copy(tb_ptr, &text_vertices[0], int(size_to_copy))
	sdl.UnmapGPUTransferBuffer(gpu, transfer_buffer)

	copy_cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)
	text_copy_pass := sdl.BeginGPUCopyPass(copy_cmd_buf)
	sdl.UploadToGPUBuffer(
		text_copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{buffer = text_gpu_vertex_buffer, offset = 0, size = u32(size_to_copy)},
		false,
	)
	sdl.EndGPUCopyPass(text_copy_pass)
	ok := sdl.SubmitGPUCommandBuffer(copy_cmd_buf)
	assert(ok)

	sdl.BindGPUGraphicsPipeline(render_pass, text_pipeline)
	for i in 0 ..< draw_count {
		cmd := draw_commands[i]
		sdl.PushGPUFragmentUniformData(cmd_buf, 0, &cmd.color, size_of(cmd.color))
		binding := sdl.GPUBufferBinding {
			buffer = text_gpu_vertex_buffer,
			offset = cmd.offset * size_of(f32),
		}
		sdl.BindGPUVertexBuffers(render_pass, 0, &binding, 1)
		sdl.DrawGPUPrimitives(render_pass, cmd.count, 1, 0, 0)
	}
}

orthographic_matrix :: proc(left, right, bottom, top, near, far: f32) -> matrix[4, 4]f32 {
	a := 2 / (right - left)
	b := 2 / (top - bottom)
	c := -2 / (far - near)
	tx := -(right + left) / (right - left)
	ty := -(top + bottom) / (top - bottom)
	tz := -(far + near) / (far - near)
	return matrix[4, 4]f32{
		a, 0, 0, tx, 
		0, b, 0, ty, 
		0, 0, c, tz, 
		0, 0, 0, 1, 
	}
}

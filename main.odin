package main

import "base:runtime"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:mem"
import sdl "vendor:sdl3"

default_context: runtime.Context

main :: proc() {
	context.logger = log.create_console_logger()
	default_context = context

	when ODIN_DEBUG {
		os_set := sdl.SetHint(sdl.HINT_RENDER_GPU_DEBUG, "1")
		_ = os_set
	}

	sdl.SetLogPriorities(.VERBOSE)
	sdl.SetLogOutputFunction(
		proc "c" (
			userdata: rawptr,
			category: sdl.LogCategory,
			priority: sdl.LogPriority,
			message: cstring,
		) {
			context = default_context
			log.debugf("SDL {} [{}]: {}", category, priority, message)
		},
		nil,
	)

	ok := sdl.SetAppMetadata("arctic char*", "0.1.0", "sh.fogo.arctic-char")
	assert(ok)

	ok = sdl.Init({.VIDEO})
	assert(ok)
	defer sdl.Quit()

	window := sdl.CreateWindow("arctic char*", 1024, 768, {.RESIZABLE})
	assert(window != nil)
	defer sdl.DestroyWindow(window)

	gpu_debug := ODIN_DEBUG
	gpu := sdl.CreateGPUDevice(shader_format, gpu_debug, nil)
	assert(gpu != nil)
	defer sdl.DestroyGPUDevice(gpu)

	ok = sdl.ClaimWindowForGPUDevice(gpu, window)
	assert(ok)
	defer sdl.ReleaseWindowFromGPUDevice(gpu, window)

	win_size: [2]i32
	ok = sdl.GetWindowSize(window, &win_size.x, &win_size.y)
	assert(ok)

	vert_shader := load_shader(gpu, vert_shader_code, .VERTEX, num_uniform_buffers = 1)
	frag_shader := load_shader(gpu, frag_shader_code, .FRAGMENT, num_uniform_buffers = 0)
	assert(vert_shader != nil)
	assert(frag_shader != nil)

	vertex_buffer_description := sdl.GPUVertexBufferDescription {
		slot               = 0,
		pitch              = size_of(VertexData),
		input_rate         = .VERTEX,
		instance_step_rate = 0,
	}
	vertex_attributes := [2]sdl.GPUVertexAttribute {
		{location = 0, buffer_slot = 0, format = .FLOAT3, offset = 0},
		{location = 1, buffer_slot = 0, format = .FLOAT4, offset = size_of(Vec3)},
	}
	target_info := sdl.GPUGraphicsPipelineTargetInfo {
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
				vertex_buffer_descriptions = &vertex_buffer_description,
				num_vertex_attributes = len(vertex_attributes),
				vertex_attributes = &vertex_attributes[0],
			},
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {
				cull_mode = .BACK,
				front_face = .COUNTER_CLOCKWISE,
			},
			target_info = target_info,
		},
	)
	assert(pipeline != nil)
	defer sdl.ReleaseGPUGraphicsPipeline(gpu, pipeline)

	sdl.ReleaseGPUShader(gpu, vert_shader)
	sdl.ReleaseGPUShader(gpu, frag_shader)

	vertices, indices := load_mesh_data("./assets/suzanne.glb")
	defer delete(vertices)
	defer delete(indices)

	vertex_data_size := len(vertices) * size_of(VertexData)
	index_data_size := len(indices) * size_of(u16)
	total_upload_size := vertex_data_size + index_data_size

	transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = u32(total_upload_size), props = 0},
	)
	assert(transfer_buffer != nil)
	defer sdl.ReleaseGPUTransferBuffer(gpu, transfer_buffer)

	transfer_ptr := transmute([^]byte)sdl.MapGPUTransferBuffer(gpu, transfer_buffer, false)
	mem.copy(transfer_ptr[:], raw_data(vertices), vertex_data_size)
	mem.copy(transfer_ptr[vertex_data_size:], raw_data(indices), index_data_size)
	sdl.UnmapGPUTransferBuffer(gpu, transfer_buffer)

	vertex_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.VERTEX}, size = u32(vertex_data_size), props = 0},
	)
	assert(vertex_buffer != nil)
	defer sdl.ReleaseGPUBuffer(gpu, vertex_buffer)

	index_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.INDEX}, size = u32(index_data_size), props = 0},
	)
	assert(index_buffer != nil)
	defer sdl.ReleaseGPUBuffer(gpu, index_buffer)

	copy_cmd := sdl.AcquireGPUCommandBuffer(gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_cmd)
	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{buffer = vertex_buffer, offset = 0, size = u32(vertex_data_size)},
		false,
	)
	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = u32(vertex_data_size)},
		{buffer = index_buffer, offset = 0, size = u32(index_data_size)},
		false,
	)
	sdl.EndGPUCopyPass(copy_pass)
	ok = sdl.SubmitGPUCommandBuffer(copy_cmd)
	assert(ok)

	log.debugf("Loaded Suzanne: vertices=%d indices=%d", len(vertices), len(indices))

	last_ticks := sdl.GetTicks()
	total_time: f32

	main_loop: for {
		new_ticks := sdl.GetTicks()
		delta_time := f32(new_ticks - last_ticks) / 1000
		last_ticks = new_ticks
		total_time += delta_time

		ev: sdl.Event
		for sdl.PollEvent(&ev) {
			#partial switch ev.type {
			case .QUIT:
				break main_loop
			case .WINDOW_RESIZED:
				ok = sdl.GetWindowSize(window, &win_size.x, &win_size.y)
				assert(ok)
			case .KEY_DOWN:
				if ev.key.scancode == .Q || ev.key.scancode == .ESCAPE {
					break main_loop
				}
			}
		}

		aspect := f32(win_size.x) / f32(win_size.y)
		model := linalg.matrix4_rotate_f32(total_time * 0.8, {0, 1, 0})
		view := linalg.matrix4_translate_f32({0, 0, -5})
		proj := linalg.matrix4_perspective_f32(linalg.to_radians(f32(70)), aspect, 0.1, 100)

		vertex_uniforms := struct {
			mvp: matrix[4, 4]f32,
		} {
			mvp = proj * view * model,
		}

		cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)
		swapchain_tex: ^sdl.GPUTexture
		ok = sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, window, &swapchain_tex, nil, nil)
		assert(ok)

		if swapchain_tex != nil {
			color_target := sdl.GPUColorTargetInfo {
				texture     = swapchain_tex,
				load_op     = .CLEAR,
				store_op    = .STORE,
				clear_color = {0.04, 0.05, 0.07, 1.0},
			}

			render_pass := sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, nil)
			sdl.BindGPUGraphicsPipeline(render_pass, pipeline)

			vertex_buffer_binding := sdl.GPUBufferBinding{buffer = vertex_buffer, offset = 0}
			index_buffer_binding := sdl.GPUBufferBinding{buffer = index_buffer, offset = 0}
			sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)
			sdl.BindGPUIndexBuffer(render_pass, index_buffer_binding, ._16BIT)
			sdl.PushGPUVertexUniformData(cmd_buf, 0, &vertex_uniforms, size_of(vertex_uniforms))

			sdl.DrawGPUIndexedPrimitives(render_pass, u32(len(indices)), 1, 0, 0, 0)
			sdl.EndGPURenderPass(render_pass)
		}

		ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
		assert(ok)
	}

	log.debug("Goodbye!")
}

@(export)
NvOptimusEnablement: u32 = 1

@(export)
AmdPowerXpressRequestHighPerformance: i32 = 1

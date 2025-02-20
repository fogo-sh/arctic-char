package main

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:strings"
import "vendor:cgltf"
import sdl "vendor:sdl3"
import "vendor:stb/easy_font"

default_context: runtime.Context

when ODIN_OS == .Darwin {
	shader_entrypoint := "main0"
	shader_format := sdl.GPUShaderFormat{.MSL}

	frag_shader_code := #load("shaders/msl/shader.msl.frag")
	vert_shader_code := #load("shaders/msl/shader.msl.vert")

	text_vert_shader_code := #load("shaders/msl/text_shader.msl.vert")
	text_frag_shader_code := #load("shaders/msl/text_shader.msl.frag")
} else when ODIN_OS == .Windows {
	shader_entrypoint := "main0"
	shader_format := sdl.GPUShaderFormat{.DXIL}

	frag_shader_code := #load("shaders/dxil/shader.dxil.frag")
	vert_shader_code := #load("shaders/dxil/shader.dxil.vert")

	text_vert_shader_code := #load("shaders/dxil/text_shader.dxil.vert")
	text_frag_shader_code := #load("shaders/dxil/text_shader.dxil.frag")
} else {
	shader_entrypoint := "main"

	shader_format := sdl.GPUShaderFormat{.SPIRV}

	frag_shader_code := #load("shaders/spv/shader.spv.frag")
	vert_shader_code := #load("shaders/spv/shader.spv.vert")

	text_vert_shader_code := #load("shaders/spv/text_shader.spv.vert")
	text_frag_shader_code := #load("shaders/spv/text_shader.spv.frag")
}

INSTANCES :: 16

UBO :: struct {
	mvp: [INSTANCES]matrix[4, 4]f32,
}

TextUBO :: struct {
	ortho: matrix[4, 4]f32,
}

TextDrawCommand :: struct {
	offset: u32,
	count:  u32,
	color:  [4]f32,
}

ModelInfo :: struct {
	offset:        u32,
	vertex_count:  u32,
	vertex_stride: u32,
}

ModelDataInfo :: struct {
	data_ptr:      ^u8,
	size:          u32,
	vertex_count:  u32,
	vertex_stride: u32,
}


main :: proc() {
	context.logger = log.create_console_logger()
	default_context = context

	ok := sdl.SetAppMetadata("arctic char*", "0.1.0", "sh.fogo.arctic-char");assert(ok)

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

	when ODIN_DEBUG {
		log.debug("Debug enabled")

		sdl.SetHint(sdl.HINT_RENDER_GPU_DEBUG, "1")
		sdl.SetHint(sdl.HINT_RENDER_DIRECT3D11_DEBUG, "1")

		gpu_debug := true
	} else {
		gpu_debug := false
	}

	ok = sdl.Init({.VIDEO, .AUDIO});assert(ok)
	defer sdl.Quit()

	window := sdl.CreateWindow("arctic char*", 512, 512, {});assert(window != nil)

	gpu := sdl.CreateGPUDevice(shader_format, gpu_debug, nil);assert(gpu != nil)

	spec: sdl.AudioSpec
	wav_data: [^]u8
	wav_data_len: u32

	ok = sdl.LoadWAV("./assets/sound.wav", &spec, &wav_data, &wav_data_len);assert(ok)

	stream := sdl.OpenAudioDeviceStream(
		sdl.AUDIO_DEVICE_DEFAULT_PLAYBACK,
		&spec,
		nil,
		nil,
	);assert(stream != nil)
	defer sdl.CloseAudioDevice(sdl.AUDIO_DEVICE_DEFAULT_PLAYBACK)

	ok = sdl.ResumeAudioStreamDevice(stream);assert(ok)

	ok = sdl.ClaimWindowForGPUDevice(gpu, window);assert(ok)

	win_size: [2]i32
	ok = sdl.GetWindowSize(window, &win_size.x, &win_size.y);assert(ok)

	depth_info := sdl.GPUTextureCreateInfo {
		type                 = sdl.GPUTextureType.D2,
		format               = sdl.GPUTextureFormat.D32_FLOAT,
		usage                = sdl.GPUTextureUsageFlags {
			sdl.GPUTextureUsageFlag.DEPTH_STENCIL_TARGET,
		},
		width                = cast(u32)win_size.x,
		height               = cast(u32)win_size.y,
		layer_count_or_depth = 1,
		num_levels           = 1,
		sample_count         = sdl.GPUSampleCount._1,
		props                = 0,
	}
	depth_texture := sdl.CreateGPUTexture(gpu, depth_info)
	assert(depth_texture != nil)
	defer sdl.ReleaseGPUTexture(gpu, depth_texture)

	vert_shader := load_shader(gpu, vert_shader_code, .VERTEX, 1)
	frag_shader := load_shader(gpu, frag_shader_code, .FRAGMENT, 0)

	mesh_vertex_buffer_description := sdl.GPUVertexBufferDescription {
		slot               = 0,
		pitch              = 28,
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
		offset      = 12,
	}
	mesh_vertex_attributes := [2]sdl.GPUVertexAttribute {
		mesh_vertex_attribute_position,
		mesh_vertex_attribute_color,
	}

	text_vertex_buffer_description := sdl.GPUVertexBufferDescription {
		slot               = 0,
		pitch              = 16,
		input_rate         = .VERTEX,
		instance_step_rate = 0,
	}
	text_vertex_attribute_description := sdl.GPUVertexAttribute {
		location    = 0,
		buffer_slot = 0,
		format      = .FLOAT4,
		offset      = 0,
	}

	depth_state := sdl.GPUDepthStencilState {
		compare_op = sdl.GPUCompareOp.LESS,
		back_stencil_state = sdl.GPUStencilOpState {
			fail_op = sdl.GPUStencilOp.KEEP,
			pass_op = sdl.GPUStencilOp.KEEP,
			depth_fail_op = sdl.GPUStencilOp.KEEP,
			compare_op = sdl.GPUCompareOp.ALWAYS,
		},
		front_stencil_state = sdl.GPUStencilOpState {
			fail_op = sdl.GPUStencilOp.KEEP,
			pass_op = sdl.GPUStencilOp.KEEP,
			depth_fail_op = sdl.GPUStencilOp.KEEP,
			compare_op = sdl.GPUCompareOp.ALWAYS,
		},
		compare_mask = 0xff,
		write_mask = 0xff,
		enable_depth_test = true,
		enable_depth_write = true,
		enable_stencil_test = false,
	}

	target_info: sdl.GPUGraphicsPipelineTargetInfo = sdl.GPUGraphicsPipelineTargetInfo {
		num_color_targets         = 1,
		color_target_descriptions = &sdl.GPUColorTargetDescription {
			format = sdl.GetGPUSwapchainTextureFormat(gpu, window),
			blend_state = sdl.GPUColorTargetBlendState{},
		},
		depth_stencil_format      = sdl.GPUTextureFormat.D32_FLOAT,
		has_depth_stencil_target  = true,
	}

	log.debug("Swapchain texture format: {}", sdl.GetGPUSwapchainTextureFormat(gpu, window))
	log.debug(
		"Target info: {num_color_targets: {}, depth_stencil_format: {}}",
		target_info.num_color_targets,
		target_info.depth_stencil_format,
	)

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		{
			vertex_shader = vert_shader,
			fragment_shader = frag_shader,
			vertex_input_state = {
				num_vertex_buffers = 1,
				vertex_buffer_descriptions = &mesh_vertex_buffer_description,
				num_vertex_attributes = 2,
				vertex_attributes = &mesh_vertex_attributes[0],
			},
			primitive_type = .TRIANGLELIST,
			rasterizer_state = sdl.GPURasterizerState {
				cull_mode = .BACK,
				front_face = .COUNTER_CLOCKWISE,
			},
			depth_stencil_state = depth_state,
			target_info = target_info,
		},
	)

	log.debug("Created GPU Pipeline")

	sdl.ReleaseGPUShader(gpu, vert_shader)
	sdl.ReleaseGPUShader(gpu, frag_shader)

	suzanne_data := load_mesh_data("./assets/suzanne.glb")
	sphere_data := load_mesh_data("./assets/sphere.glb")

	total_size := suzanne_data.size + sphere_data.size
	combined_data, err := mem.alloc(int(total_size));assert(err == nil)

	mem.copy(combined_data, suzanne_data.data_ptr, int(suzanne_data.size))

	sphere_dest_ptr := cast(^u8)(uintptr(combined_data) + uintptr(suzanne_data.size))
	mem.copy(sphere_dest_ptr, sphere_data.data_ptr, int(sphere_data.size))

	suzanne_info: ModelInfo = ModelInfo {
		offset        = 0,
		vertex_count  = suzanne_data.vertex_count,
		vertex_stride = suzanne_data.vertex_stride,
	}
	sphere_info: ModelInfo = ModelInfo {
		offset        = suzanne_data.vertex_count,
		vertex_count  = sphere_data.vertex_count,
		vertex_stride = sphere_data.vertex_stride,
	}

	transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = total_size, props = 0},
	)
	tb_ptr := sdl.MapGPUTransferBuffer(gpu, transfer_buffer, false)
	mem.copy(tb_ptr, combined_data, int(total_size))
	sdl.UnmapGPUTransferBuffer(gpu, transfer_buffer)

	global_vertex_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.VERTEX, .INDEX}, size = total_size, props = 0},
	)

	copy_command_buffer := sdl.AcquireGPUCommandBuffer(gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_command_buffer)
	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{buffer = global_vertex_buffer, offset = 0, size = total_size},
		false,
	)
	sdl.EndGPUCopyPass(copy_pass)
	ok = sdl.SubmitGPUCommandBuffer(copy_command_buffer);assert(ok)

	mem.free(combined_data)
	mem.free(suzanne_data.data_ptr)
	mem.free(sphere_data.data_ptr)

	ROTATION_SPEED := linalg.to_radians(f32(90))
	rotation := f32(0)

	proj_mat := linalg.matrix4_perspective_f32(
		linalg.to_radians(f32(70)),
		f32(win_size.x) / f32(win_size.y),
		0.1,
		1000,
	)

	last_ticks := sdl.GetTicks()

	camera_pos: [3]f32 = [3]f32{-21.3, -40.0, 25.0}
	move_forward: bool = false
	move_backward: bool = false
	move_left: bool = false
	move_right: bool = false
	move_up: bool = false
	move_down: bool = false
	shift_down: bool = false

	/*
	text_pipeline, text_gpu_vertex_buffer := setup_text_pipeline(
		gpu,
		window,
		&text_vertex_buffer_description,
		&text_vertex_attribute_description,
	)

	log.debug("Created Text Pipeline")
	*/

	text_color1: [4]f32 = [4]f32{1.0, 0.5, 0.0, 1.0}
	text_color2: [4]f32 = [4]f32{0.0, 1.0, 1.0, 1.0}

	log.debug("Ready for main loop")

	main_loop: for {
		new_ticks := sdl.GetTicks()
		delta_time := f32(new_ticks - last_ticks) / 1000
		last_ticks = new_ticks

		// events
		ev: sdl.Event
		for sdl.PollEvent(&ev) {
			#partial switch ev.type {
			case .QUIT:
				break main_loop
			case .KEY_DOWN:
				if ev.key.scancode == .ESCAPE {
					break main_loop
				} else if ev.key.scancode == .SPACE {
					text_color1 = generate_random_color()
					text_color2 = generate_random_color()
				} else if ev.key.scancode == .W {
					move_forward = true
				} else if ev.key.scancode == .S {
					move_backward = true
				} else if ev.key.scancode == .A {
					move_left = true
				} else if ev.key.scancode == .D {
					move_right = true
				} else if ev.key.scancode == .UP {
					move_up = true
				} else if ev.key.scancode == .DOWN {
					move_down = true
				} else if ev.key.scancode == .LSHIFT || ev.key.scancode == .RSHIFT {
					shift_down = true
				}
			case .KEY_UP:
				if ev.key.scancode == .W {
					move_forward = false
				} else if ev.key.scancode == .S {
					move_backward = false
				} else if ev.key.scancode == .A {
					move_left = false
				} else if ev.key.scancode == .D {
					move_right = false
				} else if ev.key.scancode == .UP {
					move_up = false
				} else if ev.key.scancode == .DOWN {
					move_down = false
				} else if ev.key.scancode == .LSHIFT || ev.key.scancode == .RSHIFT {
					shift_down = false
				}
			}
		}

		move_speed := f32(5.0)
		if shift_down {
			move_speed = f32(20.0)
		}
		dt_move := move_speed * delta_time
		if move_forward {
			camera_pos[2] -= dt_move
		}
		if move_backward {
			camera_pos[2] += dt_move
		}
		if move_left {
			camera_pos[0] -= dt_move
		}
		if move_right {
			camera_pos[0] += dt_move
		}
		if move_up {
			camera_pos[1] += dt_move
		}
		if move_down {
			camera_pos[1] -= dt_move
		}

		view_mat := linalg.matrix4_translate_f32({-camera_pos[0], -camera_pos[1], -camera_pos[2]})

		// update game state
		rotation += ROTATION_SPEED * delta_time

		ubo: UBO
		gridWidth := 32
		gridHeight := 32
		spacing := f32(2.5)

		idx: int = 0
		for i := 0; i < gridHeight; i += 1 {
			for j := 0; j < gridWidth; j += 1 {
				if idx >= INSTANCES {
					break
				}
				offset_x := (f32(j) - f32(gridWidth) / 2.0) * spacing
				offset_y := (f32(i) - f32(gridHeight) / 2.0) * spacing
				model_mat :=
					linalg.matrix4_translate_f32({offset_x, offset_y, -5}) *
					linalg.matrix4_rotate_f32(rotation, {0, 1, 0})
				ubo.mvp[idx] = proj_mat * view_mat * model_mat
				idx += 1
			}
		}

		// audio
		if sdl.GetAudioStreamAvailable(stream) < cast(i32)wav_data_len {
			sdl.PutAudioStreamData(stream, wav_data, cast(i32)wav_data_len)
		}

		// render
		cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)
		swapchain_tex: ^sdl.GPUTexture
		ok = sdl.WaitAndAcquireGPUSwapchainTexture(
			cmd_buf,
			window,
			&swapchain_tex,
			nil,
			nil,
		);assert(ok)

		if swapchain_tex != nil {
			color_target := sdl.GPUColorTargetInfo {
				texture     = swapchain_tex,
				load_op     = .CLEAR,
				clear_color = {0.2, 0.4, 0.8, 1},
				store_op    = .STORE,
			}
			depth_target: sdl.GPUDepthStencilTargetInfo = sdl.GPUDepthStencilTargetInfo {
				texture          = depth_texture,
				clear_depth      = 1.0,
				load_op          = .CLEAR,
				store_op         = .STORE,
				stencil_load_op  = .DONT_CARE,
				stencil_store_op = .DONT_CARE,
				cycle            = false,
				clear_stencil    = 0,
			}
			render_pass := sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, &depth_target)
			sdl.BindGPUGraphicsPipeline(render_pass, pipeline)
			vertex_buffer_binding := sdl.GPUBufferBinding {
				buffer = global_vertex_buffer,
				offset = 0,
			}
			sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)

			sdl.PushGPUVertexUniformData(cmd_buf, 0, &ubo, size_of(ubo))
			sdl.DrawGPUPrimitives(
				render_pass,
				suzanne_info.vertex_count,
				INSTANCES,
				suzanne_info.offset,
				0,
			)

			// Place multiple spheres into the scene.
			sphere_positions: [][3]f32 = {
				[3]f32{10, 0, 0},
				[3]f32{-10, 0, 0},
				[3]f32{0, 10, 0},
				[3]f32{0, -10, 0},
				[3]f32{10, 10, 0},
				[3]f32{-10, -10, 0},
			}
			for pos in sphere_positions {
				sphere_transform := linalg.matrix4_translate_f32(pos)
				sdl.PushGPUVertexUniformData(
					cmd_buf,
					0,
					&sphere_transform,
					size_of(sphere_transform),
				)
				sdl.DrawGPUPrimitives(
					render_pass,
					sphere_info.vertex_count,
					1,
					sphere_info.offset,
					0,
				)
			}

			/*
			{
				text_vertices: [4096]f32
				vertex_offset: u32 = 0
				draw_commands: [16]TextDrawCommand
				draw_count: u32 = 0

				accumulate_text(
					"arctic char*",
					f32(win_size.x) / 2 - 160,
					f32(win_size.y) / 2 - 160,
					text_color1,
					5.0,
					&text_vertices,
					&vertex_offset,
					&draw_commands,
					&draw_count,
				)

				accumulate_text(
					"jack & natalie",
					f32(win_size.x) / 2 - 115,
					f32(win_size.y) / 2 + 130,
					text_color2,
					3.0,
					&text_vertices,
					&vertex_offset,
					&draw_commands,
					&draw_count,
				)

				render_text(
					gpu,
					cmd_buf,
					render_pass,
					text_pipeline,
					text_gpu_vertex_buffer,
					win_size,
					&text_vertices,
					vertex_offset,
					&draw_commands,
					draw_count,
				)
			}
			*/

			sdl.EndGPURenderPass(render_pass)
		}

		ok = sdl.SubmitGPUCommandBuffer(cmd_buf);assert(ok)
	}

	log.debug("Goodbye!")
}

load_shader :: proc(
	device: ^sdl.GPUDevice,
	code: []u8,
	stage: sdl.GPUShaderStage,
	num_uniform_buffers: u32,
) -> ^sdl.GPUShader {
	return sdl.CreateGPUShader(
		device,
		{
			code_size           = len(code),
			code                = raw_data(code),
			entrypoint          = strings.clone_to_cstring(shader_entrypoint), // TODO this needs to be free'd
			format              = shader_format,
			stage               = stage,
			num_uniform_buffers = num_uniform_buffers,
		},
	)
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

load_mesh_data :: proc(model_path: cstring) -> ModelDataInfo {
	options: cgltf.options
	data, result := cgltf.parse_file(options, model_path)
	assert(result == .success)
	result = cgltf.load_buffers(options, data, model_path)
	assert(result == .success)

	mesh := data.scene.nodes[0].mesh
	primitive := mesh.primitives[0]

	pos_attr: ^cgltf.attribute = nil
	col_attr: ^cgltf.attribute = nil
	for &attr in primitive.attributes {
		#partial switch attr.type {
		case cgltf.attribute_type.position:
			pos_attr = &attr
		case cgltf.attribute_type.color:
			col_attr = &attr
		}
	}
	assert(pos_attr != nil)

	pos_accessor := pos_attr.data
	col_accessor := col_attr.data

	vertex_count := pos_accessor^.count
	num_pos_components := cgltf.num_components(pos_accessor^.type)
	num_col_components := cgltf.num_components(col_accessor^.type)
	vertex_stride := (num_pos_components + num_col_components) * size_of(f32)

	idx_accessor := primitive.indices
	index_count := idx_accessor^.count

	expanded_size := u32(index_count * vertex_stride)
	expanded_data, err := mem.alloc(int(expanded_size))
	assert(err == nil)

	pos_data_size := vertex_count * num_pos_components * size_of(f32)
	pos_buffer, err2 := mem.alloc(int(pos_data_size))
	assert(err2 == nil)
	pos_slice := mem.slice_ptr(cast(^f32)pos_buffer, int(vertex_count * num_pos_components))
	_ = cgltf.accessor_unpack_floats(
		pos_accessor,
		raw_data(pos_slice),
		uint(vertex_count) * num_pos_components,
	)

	col_data_size := vertex_count * num_col_components * size_of(f32)
	col_buffer, err3 := mem.alloc(int(col_data_size))
	assert(err3 == nil)
	col_slice := mem.slice_ptr(cast(^f32)col_buffer, int(vertex_count * num_col_components))
	_ = cgltf.accessor_unpack_floats(
		col_accessor,
		raw_data(col_slice),
		uint(vertex_count) * num_col_components,
	)

	out_slice := mem.slice_ptr(
		cast(^f32)expanded_data,
		int(index_count * (num_pos_components + num_col_components)),
	)

	for i := uint(0); i < index_count; i += 3 {
		idx0 := cgltf.accessor_read_index(idx_accessor, i)
		idx1 := cgltf.accessor_read_index(idx_accessor, i + 1)
		idx2 := cgltf.accessor_read_index(idx_accessor, i + 2)

		base := i * (num_pos_components + num_col_components)
		for j := uint(0); j < num_pos_components; j += 1 {
			out_slice[base + j] = pos_slice[idx0 * num_pos_components + j]
		}
		for j := uint(0); j < num_col_components; j += 1 {
			out_slice[base + num_pos_components + j] = col_slice[idx0 * num_col_components + j]
		}

		base = (i + 1) * (num_pos_components + num_col_components)
		for j := uint(0); j < num_pos_components; j += 1 {
			out_slice[base + j] = pos_slice[idx1 * num_pos_components + j]
		}
		for j := uint(0); j < num_col_components; j += 1 {
			out_slice[base + num_pos_components + j] = col_slice[idx1 * num_col_components + j]
		}

		base = (i + 2) * (num_pos_components + num_col_components)
		for j := uint(0); j < num_pos_components; j += 1 {
			out_slice[base + j] = pos_slice[idx2 * num_pos_components + j]
		}
		for j := uint(0); j < num_col_components; j += 1 {
			out_slice[base + num_pos_components + j] = col_slice[idx2 * num_col_components + j]
		}
	}

	mem.free(pos_buffer)
	mem.free(col_buffer)
	defer cgltf.free(data)

	return ModelDataInfo {
		data_ptr = cast(^u8)expanded_data,
		size = expanded_size,
		vertex_count = u32(index_count),
		vertex_stride = u32(vertex_stride),
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


generate_random_color :: proc() -> [4]f32 {
	r := rand.float32()
	g := rand.float32()
	b := rand.float32()
	return [4]f32{r, g, b, 1.0}
}

package main

import "base:runtime"
import "core:strings"
import "core:fmt"
import "core:log"
import "core:math/linalg"
import "core:math/rand"
import "core:os"
import sdl "vendor:sdl3"

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

	blend := sdl.GPUColorTargetBlendState {
		enable_blend            = false,
		src_color_blendfactor   = sdl.GPUBlendFactor.ONE,
		dst_color_blendfactor   = sdl.GPUBlendFactor.ZERO,
		color_blend_op          = sdl.GPUBlendOp.ADD,
		src_alpha_blendfactor   = sdl.GPUBlendFactor.ONE,
		dst_alpha_blendfactor   = sdl.GPUBlendFactor.ZERO,
		alpha_blend_op          = sdl.GPUBlendOp.ADD,
		color_write_mask        = sdl.GPUColorComponentFlags{},
		enable_color_write_mask = true,
	}

	target_info: sdl.GPUGraphicsPipelineTargetInfo = sdl.GPUGraphicsPipelineTargetInfo {
		num_color_targets         = 1,
		color_target_descriptions = &sdl.GPUColorTargetDescription {
			format = sdl.GetGPUSwapchainTextureFormat(gpu, window),
			blend_state = blend,
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
			primitive_type = sdl.GPUPrimitiveType.TRIANGLELIST,
			rasterizer_state = sdl.GPURasterizerState {
				cull_mode = sdl.GPUCullMode.BACK,
				front_face = sdl.GPUFrontFace.COUNTER_CLOCKWISE,
				depth_bias_constant_factor = 0.0,
				depth_bias_clamp = 0.0,
				depth_bias_slope_factor = 0.0,
				enable_depth_bias = false,
				enable_depth_clip = true,
			},
			multisample_state = sdl.GPUMultisampleState {
				sample_count = sdl.GPUSampleCount._1,
				sample_mask = 0xFFFFFFFF,
				enable_mask = false,
			},
			depth_stencil_state = depth_state,
			target_info = target_info,
			props = 0,
		},
	)

	log.debug("Created GPU Pipeline")

	sdl.ReleaseGPUShader(gpu, vert_shader)
	sdl.ReleaseGPUShader(gpu, frag_shader)

	mesh_vertex_buffer, mesh_vertex_count, vertex_stride := load_mesh_primitive(
		gpu,
		"./assets/suzanne.glb",
	)

	log.debug("Loaded Suzanne.glb")

	transfer_buffer := sdl.CreateGPUTransferBuffer(gpu, {usage = .UPLOAD, size = 12000, props = 0})
	tb_pointer := sdl.MapGPUTransferBuffer(gpu, transfer_buffer, false)
	(cast(^[12]f32)tb_pointer)^ = {-0.5, -0.5, 0, 1, -0.5, 0.5, 0, 1, 0.5, -0.5, 0, 1}

	sdl.UnmapGPUTransferBuffer(gpu, transfer_buffer)

	vertex_buffer := sdl.CreateGPUBuffer(gpu, {usage = {.VERTEX, .INDEX}, size = 12000, props = 0})

	copy_command_buffer := sdl.AcquireGPUCommandBuffer(gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_command_buffer)
	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{buffer = vertex_buffer, offset = 0, size = size_of([12]f32)},
		false,
	)
	sdl.EndGPUCopyPass(copy_pass)
	ok = sdl.SubmitGPUCommandBuffer(copy_command_buffer);assert(ok)

	ROTATION_SPEED := linalg.to_radians(f32(90))
	rotation := f32(0)

	proj_mat := linalg.matrix4_perspective_f32(
		linalg.to_radians(f32(70)),
		f32(win_size.x) / f32(win_size.y),
		0.1,
		1000,
	)

	last_ticks := sdl.GetTicks()

	camera_pos: [3]f32 = [3]f32{0.0, -21.0, 60.0}
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
				buffer = mesh_vertex_buffer,
				offset = 0,
			}
			sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)

			sdl.PushGPUVertexUniformData(cmd_buf, 0, &ubo, size_of(ubo))
			sdl.DrawGPUPrimitives(render_pass, mesh_vertex_count, INSTANCES, 0, 0)

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
			code_size = len(code),
			code = raw_data(code),
			entrypoint = strings.clone_to_cstring(shader_entrypoint), // TODO this needs to be free'd
			format = shader_format,
			stage = stage,
			num_uniform_buffers = num_uniform_buffers,
		},
	)
}

generate_random_color :: proc() -> [4]f32 {
	r := rand.float32()
	g := rand.float32()
	b := rand.float32()
	return [4]f32{r, g, b, 1.0}
}

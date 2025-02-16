package main

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import sdl "vendor:sdl3"
import "vendor:stb/easy_font"

default_context: runtime.Context

frag_shader_code := #load("shaders/msl/shader.msl.frag")
vert_shader_code := #load("shaders/msl/shader.msl.vert")

text_vert_shader_code := #load("shaders/msl/text_shader.msl.vert")
text_frag_shader_code := #load("shaders/msl/text_shader.msl.frag")

TextUBO :: struct {
	ortho: matrix[4, 4]f32,
}

UBO :: struct {
	mvp: matrix[4, 4]f32,
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

	ok = sdl.Init({.VIDEO, .AUDIO});assert(ok)
	defer sdl.Quit()

	window := sdl.CreateWindow("arctic char*", 512, 512, {});assert(window != nil)

	gpu := sdl.CreateGPUDevice({.MSL}, true, nil);assert(gpu != nil)

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

	vert_shader := load_shader(gpu, vert_shader_code, .VERTEX, 1)
	frag_shader := load_shader(gpu, frag_shader_code, .FRAGMENT, 0)

	text_vert_shader := load_shader(gpu, text_vert_shader_code, .VERTEX, 1)
	text_frag_shader := load_shader(gpu, text_frag_shader_code, .FRAGMENT, 1)

	/*
	GPUVertexInputState :: struct {
		vertex_buffer_descriptions: [^]GPUVertexBufferDescription `fmt:"v,num_vertex_buffers"`,    /**< A pointer to an array of vertex buffer descriptions. */
		num_vertex_buffers:         Uint32,                                                        /**< The number of vertex buffer descriptions in the above array. */
		vertex_attributes:          [^]GPUVertexAttribute         `fmt:"v,num_vertex_attributes"`, /**< A pointer to an array of vertex attribute descriptions. */
		num_vertex_attributes:      Uint32,                                                        /**< The number of vertex attribute descriptions in the above array. */
	}
	*/
	vertex_buffer_description := sdl.GPUVertexBufferDescription {
		slot               = 0,
		pitch              = 16,
		input_rate         = .VERTEX,
		instance_step_rate = 0, // THIS IS IGNORED!!!! NEEDED FOR TYPING
	}
	vertex_attribute_description := sdl.GPUVertexAttribute {
		location    = 0,
		buffer_slot = 0,
		format      = .FLOAT4,
		offset      = 0,
	}
	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		{
			vertex_shader = vert_shader,
			fragment_shader = frag_shader,
			vertex_input_state = {
				num_vertex_buffers = 1,
				vertex_buffer_descriptions = &vertex_buffer_description,
				num_vertex_attributes = 1,
				vertex_attributes = &vertex_attribute_description,
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

	sdl.ReleaseGPUShader(gpu, vert_shader)
	sdl.ReleaseGPUShader(gpu, frag_shader)

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

	win_size: [2]i32
	ok = sdl.GetWindowSize(window, &win_size.x, &win_size.y);assert(ok)

	ROTATION_SPEED := linalg.to_radians(f32(90))
	rotation := f32(0)

	proj_mat := linalg.matrix4_perspective_f32(
		linalg.to_radians(f32(70)),
		f32(win_size.x) / f32(win_size.y),
		0.0001,
		1000,
	)

	last_ticks := sdl.GetTicks()

	text_pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		{
			vertex_shader = text_vert_shader,
			fragment_shader = text_frag_shader,
			vertex_input_state = {
				num_vertex_buffers = 1,
				vertex_buffer_descriptions = &vertex_buffer_description,
				num_vertex_attributes = 1,
				vertex_attributes = &vertex_attribute_description,
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

	text_gpu_vertex_buffer := sdl.CreateGPUBuffer(gpu, {usage = {.VERTEX}, size = 4096, props = 0})
	text_transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = 4096, props = 0},
	)

	text_color1: [4]f32 = [4]f32{1.0, 0.5, 0.0, 1.0}
	text_color2: [4]f32 = [4]f32{0.0, 1.0, 1.0, 1.0}

	main_loop: for {
		new_ticks := sdl.GetTicks()
		delta_time := f32(new_ticks - last_ticks) / 1000
		last_ticks = new_ticks

		// process events
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
				}
			}
		}

		// update game state

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

		rotation += ROTATION_SPEED * delta_time
		model_mat :=
			linalg.matrix4_translate_f32({0, 0, -5}) *
			linalg.matrix4_rotate_f32(rotation, {0, 1, 0})

		ubo := UBO {
			mvp = proj_mat * model_mat,
		}

		if swapchain_tex != nil {
			color_target := sdl.GPUColorTargetInfo {
				texture     = swapchain_tex,
				load_op     = .CLEAR,
				clear_color = {0.2, 0.4, 0.8, 1},
				store_op    = .STORE,
			}
			render_pass := sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, nil)
			sdl.BindGPUGraphicsPipeline(render_pass, pipeline)
			vertex_buffer_binding := sdl.GPUBufferBinding {
				buffer = vertex_buffer,
				offset = 0,
			}
			sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)
			sdl.PushGPUVertexUniformData(cmd_buf, 0, &ubo, size_of(ubo))
			sdl.DrawGPUPrimitives(render_pass, 3, 1, 0, 0)

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
				mem.copy(tb_ptr, &text_vertices, int(size_to_copy))
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

			sdl.EndGPURenderPass(render_pass)
		}

		ok = sdl.SubmitGPUCommandBuffer(cmd_buf);assert(ok)
	}
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
			entrypoint = "main0",
			format = {.MSL},
			stage = stage,
			num_uniform_buffers = num_uniform_buffers,
		},
	)
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

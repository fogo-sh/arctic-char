package main

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:os"
import "vendor:cgltf"
import sdl "vendor:sdl3"

default_context: runtime.Context

frag_shader_code := #load("shaders/msl/shader.msl.frag")
vert_shader_code := #load("shaders/msl/shader.msl.vert")

UBO :: struct {
	mvp: matrix[4, 4]f32,
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

	mesh_vertex_buffer_description := sdl.GPUVertexBufferDescription {
		slot               = 0,
		pitch              = 12,
		input_rate         = .VERTEX,
		instance_step_rate = 0,
	}
	mesh_vertex_attribute_description := sdl.GPUVertexAttribute {
		location    = 0,
		buffer_slot = 0,
		format      = .FLOAT3,
		offset      = 0,
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

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		{
			vertex_shader = vert_shader,
			fragment_shader = frag_shader,
			vertex_input_state = {
				num_vertex_buffers = 1,
				vertex_buffer_descriptions = &mesh_vertex_buffer_description,
				num_vertex_attributes = 1,
				vertex_attributes = &mesh_vertex_attribute_description,
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

	{
		model_path: cstring = "./assets/suzanne.glb"
		options: cgltf.options
		data, result := cgltf.parse_file(options, model_path)
		assert(result == .success)

		result = cgltf.load_buffers(options, data, model_path)
		assert(result == .success)

		for node in data.scene.nodes {
			mesh := node.mesh
			primitive := mesh.primitives[0]

			fmt.printf("primitive type: %v\n", primitive.type)
			fmt.printf("primitive indices: %v\n", primitive.indices)
			fmt.printf("primitive attributes count: %v\n", len(primitive.attributes))
			for attr in primitive.attributes {
				fmt.printf("  attribute type: %v, accessor type: %v\n", attr.type, attr.data.type)
			}
		}

		defer cgltf.free(data)
	}

	mesh_vertex_buffer, mesh_vertex_count, vertex_stride := load_mesh_primitive(
		gpu,
		"./assets/suzanne.glb",
	)
	log.debugf(
		"Mesh loaded: vertex_count = %d, vertex_stride = %d",
		mesh_vertex_count,
		vertex_stride,
	)

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

	text_pipeline, text_gpu_vertex_buffer := setup_text_pipeline(
		gpu,
		window,
		&text_vertex_buffer_description,
		&text_vertex_attribute_description,
	)

	text_color1: [4]f32 = [4]f32{1.0, 0.5, 0.0, 1.0}
	text_color2: [4]f32 = [4]f32{0.0, 1.0, 1.0, 1.0}

	main_loop: for {
		new_ticks := sdl.GetTicks()
		delta_time := f32(new_ticks - last_ticks) / 1000
		last_ticks = new_ticks
		log.debugf("Frame start - new_ticks: %d, delta_time: %f", new_ticks, delta_time)

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
		rotation += ROTATION_SPEED * delta_time
		log.debugf("Updated rotation value: %f", rotation)

		model_mat :=
			linalg.matrix4_translate_f32({0, 0, -5}) *
			linalg.matrix4_rotate_f32(rotation, {0, 1, 0})
		ubo := UBO {
			mvp = proj_mat * model_mat,
		}
		log.debugf("Computed MVP matrix: %v", ubo.mvp)

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
			render_pass := sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, nil)
			sdl.BindGPUGraphicsPipeline(render_pass, pipeline)
			vertex_buffer_binding := sdl.GPUBufferBinding {
				buffer = mesh_vertex_buffer,
				offset = 0,
			}
			sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)
			sdl.PushGPUVertexUniformData(cmd_buf, 0, &ubo, size_of(ubo))
			log.debugf("Pushed UBO data to GPU: %v", ubo.mvp)
			sdl.DrawGPUPrimitives(render_pass, mesh_vertex_count, 1, 0, 0)

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

generate_random_color :: proc() -> [4]f32 {
	r := rand.float32()
	g := rand.float32()
	b := rand.float32()
	return [4]f32{r, g, b, 1.0}
}

load_mesh_primitive :: proc(
	gpu: ^sdl.GPUDevice,
	model_path: cstring,
) -> (
	^sdl.GPUBuffer,
	u32,
	u32,
) {
	options: cgltf.options
	data, result := cgltf.parse_file(options, model_path)
	assert(result == .success)
	result = cgltf.load_buffers(options, data, model_path)
	assert(result == .success)

	mesh := data.scene.nodes[0].mesh
	primitive := mesh.primitives[0]

	pos_attr: ^cgltf.attribute = nil
	for &attr in primitive.attributes {
		if attr.type == cgltf.attribute_type.position {
			pos_attr = &attr
			break
		}
	}
	assert(pos_attr != nil)
	pos_accessor := pos_attr.data

	vertex_count: u32 = u32(pos_accessor^.count)
	num_components := cgltf.num_components(pos_accessor^.type)
	vertex_stride := num_components * size_of(f32)
	data_size := vertex_count * u32(vertex_stride)

	cpu_vertex_data: rawptr = nil

	idx_accessor := primitive.indices
	index_count := idx_accessor^.count
	expanded_size := index_count * vertex_stride

	expanded_data, err := mem.alloc(int(expanded_size))
	assert(err == nil)
	orig_buffer, other_err := mem.alloc(int(data_size))
	assert(other_err == nil)
	_ = cgltf.accessor_unpack_floats(
		pos_accessor,
		cast([^]f32)orig_buffer,
		uint(vertex_count) * num_components,
	)
	orig := cast(^f32)orig_buffer
	expanded := cast(^f32)expanded_data

	expanded_slice := mem.slice_ptr(expanded, int(index_count * num_components))
	orig_slice := mem.slice_ptr(orig, int(uint(vertex_count) * num_components))
	for i := uint(0); i < index_count; i += 1 {
		idx := cgltf.accessor_read_index(idx_accessor, i)
		for j := uint(0); j < num_components; j += 1 {
			expanded_slice[i * num_components + j] = orig_slice[idx * num_components + j]
		}
	}
	mem.free(orig_buffer)
	cpu_vertex_data = expanded_data
	vertex_count = u32(index_count)
	data_size = u32(expanded_size)

	if index_count > 0 {
		sample_count := 5
		log.debugf("First %d expanded vertices:", sample_count)
		for i := uint(0); int(i) < sample_count; i += 1 {
			v0 := expanded_slice[i * num_components + 0]
			v1 := expanded_slice[i * num_components + 1]
			v2 := expanded_slice[i * num_components + 2]
			log.debugf("  Vertex %d: (%f, %f, %f)", i, v0, v1, v2)
		}
	}

	transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = data_size, props = 0},
	)
	tb_ptr := sdl.MapGPUTransferBuffer(gpu, transfer_buffer, false)
	mem.copy(tb_ptr, cpu_vertex_data, int(data_size))
	sdl.UnmapGPUTransferBuffer(gpu, transfer_buffer)

	vertex_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.VERTEX, .INDEX}, size = data_size, props = 0},
	)
	copy_command_buffer := sdl.AcquireGPUCommandBuffer(gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_command_buffer)
	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{buffer = vertex_buffer, offset = 0, size = data_size},
		false,
	)
	sdl.EndGPUCopyPass(copy_pass)
	ok := sdl.SubmitGPUCommandBuffer(copy_command_buffer);assert(ok)

	mem.free(cpu_vertex_data)
	defer cgltf.free(data)

	return vertex_buffer, u32(vertex_count), u32(vertex_stride)
}

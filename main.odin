package main

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:reflect"
import "core:strings"
import "vendor:cgltf"
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

Vec3 :: [3]f32

VertexData :: struct {
	pos:   Vec3,
	color: sdl.FColor,
	uv:    [2]f32,
}

ModelData :: struct {
	vertices: []VertexData,
	indices:  []u16,
}

ModelInfo :: struct {
	offset:      int,
	index_count: int,
}

Model :: enum {
	Suzanne,
	Sphere,
}

SceneObject :: struct {
	vertex_offset: int,
	vertex_count:  int,
	local_model:   matrix[4, 4]f32,
}

Movement :: struct {
	forward:  bool,
	backward: bool,
	left:     bool,
	right:    bool,
	up:       bool,
	down:     bool,
	shift:    bool,
}

main :: proc() {
	context.logger = log.create_console_logger()
	default_context = context

	ok := sdl.SetAppMetadata("arctic char*", "0.1.0", "sh.fogo.arctic-char");assert(ok)

	// -- initial setup --

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

	// -- end initial setup --

	// -- audio setup --

	ok = sdl.Init({.VIDEO, .AUDIO});assert(ok)
	defer sdl.Quit()

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

	// -- end audio setup --

	// -- window and gpu setup --

	window := sdl.CreateWindow("arctic char*", 512, 512, {});assert(window != nil)

	gpu := sdl.CreateGPUDevice(shader_format, gpu_debug, nil);assert(gpu != nil)

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
		offset      = 12,
	}
	mesh_vertex_attribute_uv := sdl.GPUVertexAttribute {
		location    = 2,
		buffer_slot = 0,
		format      = .FLOAT2,
		offset      = 28,
	}
	mesh_vertex_attributes := [3]sdl.GPUVertexAttribute {
		mesh_vertex_attribute_position,
		mesh_vertex_attribute_color,
		mesh_vertex_attribute_uv,
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

	combined_vertex_data: []VertexData
	combined_index_data: []u16

	combined_vertex_data = make(
		[]VertexData,
		len(suzanne_data.vertices) + len(sphere_data.vertices),
	)
	combined_index_data = make([]u16, len(suzanne_data.indices) + len(sphere_data.indices))

	copy(combined_vertex_data[:len(suzanne_data.vertices)], suzanne_data.vertices)
	copy(combined_index_data[:len(suzanne_data.indices)], suzanne_data.indices)

	copy(combined_vertex_data[len(suzanne_data.vertices):], sphere_data.vertices)
	copy(combined_index_data[len(suzanne_data.indices):], sphere_data.indices)

	combined_vertex_data_size := len(combined_vertex_data) * size_of(VertexData)
	combined_index_data_size := len(combined_index_data) * size_of(u16)
	total_size := combined_vertex_data_size + combined_index_data_size

	suzanne_info: ModelInfo = ModelInfo {
		offset      = 0,
		index_count = len(suzanne_data.indices),
	}
	sphere_info: ModelInfo = ModelInfo {
		offset      = suzanne_info.index_count,
		index_count = len(sphere_data.indices),
	}

	transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = u32(total_size), props = 0},
	)
	transfer_buffer_ptr := transmute([^]byte)sdl.MapGPUTransferBuffer(gpu, transfer_buffer, false)

	mem.copy(
		transfer_buffer_ptr,
		raw_data(combined_vertex_data),
		len(combined_vertex_data) * size_of(VertexData),
	)
	mem.copy(
		transfer_buffer_ptr[combined_vertex_data_size:],
		raw_data(combined_index_data),
		len(combined_index_data) * size_of(u16),
	)

	sdl.UnmapGPUTransferBuffer(gpu, transfer_buffer)

	vertex_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.VERTEX}, size = u32(combined_vertex_data_size), props = 0},
	)

	index_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.INDEX}, size = u32(combined_index_data_size), props = 0},
	)

	copy_command_buffer := sdl.AcquireGPUCommandBuffer(gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_command_buffer)
	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{buffer = vertex_buffer, offset = 0, size = u32(combined_vertex_data_size)},
		false,
	)
	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = u32(combined_vertex_data_size)},
		{buffer = index_buffer, offset = 0, size = u32(combined_index_data_size)},
		false,
	)
	sdl.EndGPUCopyPass(copy_pass)
	ok = sdl.SubmitGPUCommandBuffer(copy_command_buffer);assert(ok)

	sdl.ReleaseGPUTransferBuffer(gpu, transfer_buffer)

	// -- end window and gpu setup --

	// -- main loop setup --

	// TODO we're done with holding the model data in memory
	// we should release the memory now

	ROTATION_SPEED := linalg.to_radians(f32(90 * 10))
	rotation := f32(0)

	proj_mat := linalg.matrix4_perspective_f32(
		linalg.to_radians(f32(70)),
		f32(win_size.x) / f32(win_size.y),
		0.1,
		1000,
	)

	last_ticks := sdl.GetTicks()

	camera_pos: [3]f32 = [3]f32{0.0, 0.0, 10.0}

	movement := Movement{}

	text_color1: [4]f32 = [4]f32{1.0, 0.5, 0.0, 1.0}
	text_color2: [4]f32 = [4]f32{0.0, 1.0, 1.0, 1.0}

	log.debug("Ready for main loop")

	make_scene_object :: proc(model_type: ModelInfo, position: [3]f32) -> SceneObject {
		return SceneObject {
			vertex_offset = model_type.offset,
			vertex_count = model_type.index_count,
			local_model = linalg.matrix4_translate_f32(position),
		}
	}

	scene_objects: [8]SceneObject = {
		make_scene_object(suzanne_info, {-4.0, 1.5, 0.0}),
		make_scene_object(sphere_info, {-2.0, 1.5, 0.0}),
		make_scene_object(suzanne_info, {2.0, 1.5, 0.0}),
		make_scene_object(sphere_info, {4.0, 1.5, 0.0}),
		make_scene_object(sphere_info, {-4.0, -1.5, 0.0}),
		make_scene_object(suzanne_info, {-2.0, -1.5, 0.0}),
		make_scene_object(sphere_info, {2.0, -1.5, 0.0}),
		make_scene_object(suzanne_info, {4.0, -1.5, 0.0}),
	}

	main_loop: for {
		new_ticks := sdl.GetTicks()
		delta_time := f32(new_ticks - last_ticks) / 1000
		last_ticks = new_ticks

		// -- events --
		ev: sdl.Event
		for sdl.PollEvent(&ev) {
			#partial switch ev.type {
			case .QUIT:
				break main_loop
			case .KEY_DOWN:
				if ev.key.scancode == .ESCAPE {
					break main_loop
				} else if ev.key.scancode == .W {
					movement.forward = true
				} else if ev.key.scancode == .S {
					movement.backward = true
				} else if ev.key.scancode == .A {
					movement.left = true
				} else if ev.key.scancode == .D {
					movement.right = true
				} else if ev.key.scancode == .UP {
					movement.up = true
				} else if ev.key.scancode == .DOWN {
					movement.down = true
				} else if ev.key.scancode == .LSHIFT || ev.key.scancode == .RSHIFT {
					movement.shift = true
				}
			case .KEY_UP:
				if ev.key.scancode == .W {
					movement.forward = false
				} else if ev.key.scancode == .S {
					movement.backward = false
				} else if ev.key.scancode == .A {
					movement.left = false
				} else if ev.key.scancode == .D {
					movement.right = false
				} else if ev.key.scancode == .UP {
					movement.up = false
				} else if ev.key.scancode == .DOWN {
					movement.down = false
				} else if ev.key.scancode == .LSHIFT || ev.key.scancode == .RSHIFT {
					movement.shift = false
				}
			}
		}

		move_speed := f32(5.0)
		if movement.shift {
			move_speed = f32(20.0)
		}
		dt_move := move_speed * delta_time
		if movement.forward {
			camera_pos[2] -= dt_move
		}
		if movement.backward {
			camera_pos[2] += dt_move
		}
		if movement.left {
			camera_pos[0] -= dt_move
		}
		if movement.right {
			camera_pos[0] += dt_move
		}
		if movement.up {
			camera_pos[1] += dt_move
		}
		if movement.down {
			camera_pos[1] -= dt_move
		}

		view_mat := linalg.matrix4_translate_f32({-camera_pos[0], -camera_pos[1], -camera_pos[2]})

		rotation += ROTATION_SPEED * delta_time

		// -- audio --
		if sdl.GetAudioStreamAvailable(stream) < cast(i32)wav_data_len {
			sdl.PutAudioStreamData(stream, wav_data, cast(i32)wav_data_len)
		}

		// -- render --
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
				buffer = vertex_buffer,
				offset = 0,
			}
			index_buffer_binding := sdl.GPUBufferBinding {
				buffer = index_buffer,
				offset = 0,
			}
			sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)
			sdl.BindGPUIndexBuffer(render_pass, index_buffer_binding, ._16BIT)

			for obj in scene_objects {
				local_spin := linalg.matrix4_rotate_f32(rotation, {0, 1, 0})
				object_model := obj.local_model * local_spin
				mvp := proj_mat * view_mat * object_model

				sdl.PushGPUVertexUniformData(cmd_buf, 0, &mvp, size_of(mvp))
				sdl.DrawGPUIndexedPrimitives(
					render_pass,
					u32(obj.vertex_offset),
					u32(obj.vertex_count),
					0,
					0,
					0,
				)
			}

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

load_mesh_data :: proc(model_path: cstring) -> ModelData {
	options: cgltf.options
	data, result := cgltf.parse_file(options, model_path)
	assert(result == .success)
	result = cgltf.load_buffers(options, data, model_path)
	assert(result == .success)
	defer cgltf.free(data)

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
	idx_accessor := primitive.indices

	vertex_count := pos_accessor.count
	index_count := idx_accessor.count
	num_pos_components := cgltf.num_components(pos_accessor.type)
	num_col_components := cgltf.num_components(col_accessor.type)

	positions := make([]f32, vertex_count * num_pos_components)
	_ = cgltf.accessor_unpack_floats(
		pos_accessor,
		raw_data(positions),
		uint(vertex_count * num_pos_components),
	)

	colors := make([]f32, vertex_count * num_col_components)
	_ = cgltf.accessor_unpack_floats(
		col_accessor,
		raw_data(colors),
		uint(vertex_count * num_col_components),
	)

	vertices := make([]VertexData, vertex_count)
	for i := 0; i < int(vertex_count); i += 1 {
		pos_idx := i * int(num_pos_components)
		col_idx := i * int(num_col_components)

		vertices[i] = VertexData {
			pos   = Vec3{positions[pos_idx + 0], positions[pos_idx + 1], positions[pos_idx + 2]},
			color = sdl.FColor{colors[col_idx + 0], colors[col_idx + 1], colors[col_idx + 2], 1.0},
			uv    = [2]f32{0, 0},
		}
	}

	indices := make([]u16, index_count)
	for i := 0; i < int(index_count); i += 1 {
		indices[i] = u16(cgltf.accessor_read_index(idx_accessor, uint(i)))
	}

	return ModelData{indices = indices, vertices = vertices}
}

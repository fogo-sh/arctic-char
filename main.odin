package main

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:reflect"
import "core:strings"
import "vendor:cgltf"
import sdl "vendor:sdl3"
import stbi "vendor:stb/image"

default_context: runtime.Context

when ODIN_OS == .Darwin {
	shader_entrypoint := "main0"
	shader_format := sdl.GPUShaderFormat{.MSL}

	frag_shader_code := #load("shaders/msl/shader.msl.frag")
	vert_shader_code := #load("shaders/msl/shader.msl.vert")
} else when ODIN_OS == .Windows {
	shader_entrypoint := "main0"
	shader_format := sdl.GPUShaderFormat{.DXIL}

	frag_shader_code := #load("shaders/dxil/shader.dxil.frag")
	vert_shader_code := #load("shaders/dxil/shader.dxil.vert")
} else {
	shader_entrypoint := "main"

	shader_format := sdl.GPUShaderFormat{.SPIRV}

	frag_shader_code := #load("shaders/spv/shader.spv.frag")
	vert_shader_code := #load("shaders/spv/shader.spv.vert")
}

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

model_info_lookup: map[Model]ModelInfo

Model :: enum {
	Suzanne,
	Sphere,
	Plane,
	Reference,
}

MODEL_COUNT :: 4

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

	ok := sdl.SetAppMetadata("arctic char*", "0.1.0", "sh.fogo.arctic-char")
	assert(ok)

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

	ok = sdl.Init({.VIDEO, .AUDIO})
	assert(ok)
	defer sdl.Quit()

	spec: sdl.AudioSpec
	wav_data: [^]u8
	wav_data_len: u32

	ok = sdl.LoadWAV("./assets/sound.wav", &spec, &wav_data, &wav_data_len)
	assert(ok)

	stream := sdl.OpenAudioDeviceStream(sdl.AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nil, nil)
	assert(stream != nil)
	defer sdl.CloseAudioDevice(sdl.AUDIO_DEVICE_DEFAULT_PLAYBACK)

	ok = sdl.ResumeAudioStreamDevice(stream)
	assert(ok)

	// -- end audio setup --

	// -- window and gpu setup --

	window := sdl.CreateWindow("arctic char*", 512, 512, {})
	assert(window != nil)

	gpu := sdl.CreateGPUDevice(shader_format, gpu_debug, nil)
	assert(gpu != nil)

	ok = sdl.ClaimWindowForGPUDevice(gpu, window)
	assert(ok)

	win_size: [2]i32
	ok = sdl.GetWindowSize(window, &win_size.x, &win_size.y)
	assert(ok)

	msaa_color_texture := sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = sdl.GPUTextureType.D2,
			format = sdl.GetGPUSwapchainTextureFormat(gpu, window),
			usage = sdl.GPUTextureUsageFlags{.COLOR_TARGET},
			width = cast(u32)win_size.x,
			height = cast(u32)win_size.y,
			layer_count_or_depth = 1,
			num_levels = 1,
			sample_count = sdl.GPUSampleCount._4,
		},
	)
	assert(msaa_color_texture != nil)
	defer sdl.ReleaseGPUTexture(gpu, msaa_color_texture)

	msaa_depth_texture := sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = sdl.GPUTextureType.D2,
			format = sdl.GPUTextureFormat.D32_FLOAT,
			usage = sdl.GPUTextureUsageFlags{.DEPTH_STENCIL_TARGET},
			width = cast(u32)win_size.x,
			height = cast(u32)win_size.y,
			layer_count_or_depth = 1,
			num_levels = 1,
			sample_count = sdl.GPUSampleCount._4,
		},
	)
	assert(msaa_depth_texture != nil)
	defer sdl.ReleaseGPUTexture(gpu, msaa_depth_texture)

	vert_shader := load_shader(
		gpu,
		vert_shader_code,
		.VERTEX,
		num_uniform_buffers = 1,
		num_samplers = 0,
	)
	frag_shader := load_shader(
		gpu,
		frag_shader_code,
		.FRAGMENT,
		num_uniform_buffers = 0,
		num_samplers = 1,
	)

	img_size: [2]i32
	pixels := stbi.load("./assets/test_albedo.png", &img_size.x, &img_size.y, nil, 4)
	assert(pixels != nil)
	pixels_byte_size := int(img_size.x * img_size.y * 4)

	texture := sdl.CreateGPUTexture(
		gpu,
		{
			format = .R8G8B8A8_UNORM,
			usage = {.SAMPLER},
			width = u32(img_size.x),
			height = u32(img_size.y),
			layer_count_or_depth = 1,
			num_levels = 1,
		},
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
			multisample_state = sdl.GPUMultisampleState {
				sample_count = sdl.GPUSampleCount._4,
				sample_mask = 0xffffffff,
				enable_mask = true,
			},
			depth_stencil_state = depth_state,
			target_info = target_info,
		},
	)

	log.debug("Created GPU Pipeline")

	sdl.ReleaseGPUShader(gpu, vert_shader)
	sdl.ReleaseGPUShader(gpu, frag_shader)

	sampler := sdl.CreateGPUSampler(gpu, {})

	model_names := reflect.enum_field_names(Model)

	vertex_datas: [MODEL_COUNT][]VertexData
	index_datas: [MODEL_COUNT][]u16

	combined_vertex_count: int
	combined_index_count: int

	for i := 0; i < len(model_names); i += 1 {
		model_name := model_names[i]
		model_path := fmt.tprintf("./assets/%s.glb", model_name)

		vertex_data, index_data := load_mesh_data(model_path)

		for j in 0 ..< len(index_data) {
			index_data[j] += u16(combined_vertex_count)
		}

		vertex_datas[i] = vertex_data
		index_datas[i] = index_data

		model_enum, ok := reflect.enum_from_name(Model, model_name)
		assert(ok)

		model_info_lookup[model_enum] = ModelInfo {
			offset      = combined_index_count,
			index_count = len(index_data),
		}

		combined_vertex_count += len(vertex_data)
		combined_index_count += len(index_data)
	}

	combined_vertex_data := make([]VertexData, combined_vertex_count)
	combined_index_data := make([]u16, combined_index_count)

	vertex_data_offset := 0
	index_data_offset := 0
	for i in 0 ..< len(vertex_datas) {
		copy(combined_vertex_data[vertex_data_offset:], vertex_datas[i])
		copy(combined_index_data[index_data_offset:], index_datas[i])

		vertex_data_offset += len(vertex_datas[i])
		index_data_offset += len(index_datas[i])
	}

	total_size_in_bytes :=
		combined_vertex_count * size_of(VertexData) +
		combined_index_count * size_of(u16) +
		pixels_byte_size

	transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = u32(total_size_in_bytes), props = 0},
	)
	transfer_buffer_ptr := transmute([^]byte)sdl.MapGPUTransferBuffer(gpu, transfer_buffer, false)

	transfer_buffer_offset := 0

	mem.copy(
		transfer_buffer_ptr[transfer_buffer_offset:],
		raw_data(combined_vertex_data),
		len(combined_vertex_data) * size_of(VertexData),
	)
	transfer_buffer_offset += len(combined_vertex_data) * size_of(VertexData)

	mem.copy(
		transfer_buffer_ptr[transfer_buffer_offset:],
		raw_data(combined_index_data),
		len(combined_index_data) * size_of(u16),
	)
	transfer_buffer_offset += len(combined_index_data) * size_of(u16)

	mem.copy(transfer_buffer_ptr[transfer_buffer_offset:], pixels, pixels_byte_size)

	sdl.UnmapGPUTransferBuffer(gpu, transfer_buffer)

	vertex_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.VERTEX}, size = u32(combined_vertex_count * size_of(VertexData)), props = 0},
	)

	index_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.INDEX}, size = u32(combined_index_count * size_of(u16)), props = 0},
	)

	copy_command_buffer := sdl.AcquireGPUCommandBuffer(gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_command_buffer)

	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{
			buffer = vertex_buffer,
			offset = 0,
			size = u32(combined_vertex_count * size_of(VertexData)),
		},
		false,
	)

	buffer_offset := u32(combined_vertex_count * size_of(VertexData))

	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = buffer_offset},
		{buffer = index_buffer, offset = 0, size = u32(combined_index_count * size_of(u16))},
		false,
	)

	buffer_offset += u32(combined_index_count * size_of(u16))

	sdl.UploadToGPUTexture(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = buffer_offset},
		{texture = texture, w = u32(img_size.x), h = u32(img_size.y), d = 1},
		false,
	)

	sdl.EndGPUCopyPass(copy_pass)
	ok = sdl.SubmitGPUCommandBuffer(copy_command_buffer)
	assert(ok)

	sdl.ReleaseGPUTransferBuffer(gpu, transfer_buffer)

	// -- end window and gpu setup --

	// -- main loop setup --

	delete(combined_vertex_data)
	delete(combined_index_data)

	ROTATION_SPEED := linalg.to_radians(f32(90))
	rotation := f32(0)

	proj_mat := linalg.matrix4_perspective_f32(
		linalg.to_radians(f32(70)),
		f32(win_size.x) / f32(win_size.y),
		0.1,
		1000,
	)

	last_ticks := sdl.GetTicks()

	// Camera position and mouse-look angles
	camera_pos: [3]f32 = [3]f32{0.0, 0.0, 10.0}
	camera_yaw: f32 = 0.0
	camera_pitch: f32 = 0.0
	mouse_locked := false
	mouse_sensitivity := 0.003 // tweak as desired

	movement := Movement{}

	log.debug("Ready for main loop")

	make_scene_object :: proc(model: Model, position: [3]f32) -> SceneObject {
		model_type := model_info_lookup[model]
		return SceneObject {
			vertex_offset = model_type.offset,
			vertex_count = model_type.index_count,
			local_model = linalg.matrix4_translate_f32(position),
		}
	}

	scene_objects: []SceneObject = {make_scene_object(Model.Reference, {0.0, 0.0, 0.0})}

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

			case .MOUSE_BUTTON_DOWN:
				if ev.button.button == 1 {
					ok = sdl.SetWindowRelativeMouseMode(window, true)
					assert(ok)
					mouse_locked = true
				}

			case .KEY_DOWN:
				if ev.key.scancode == .Q {
					break main_loop
				} else if ev.key.scancode == .ESCAPE {
					ok = sdl.SetWindowRelativeMouseMode(window, false)
					assert(ok)
					mouse_locked = false
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

			case .MOUSE_MOTION:
				if mouse_locked {
					camera_yaw -= ev.motion.xrel * f32(mouse_sensitivity)
					camera_pitch -= ev.motion.yrel * f32(mouse_sensitivity)

					if camera_pitch > linalg.to_radians(f32(89)) {
						camera_pitch = linalg.to_radians(f32(89))
					} else if camera_pitch < linalg.to_radians(f32(-89)) {
						camera_pitch = linalg.to_radians(f32(-89))
					}
				}
			}
		}

		sin_y := math.sin(camera_yaw)
		cos_y := math.cos(camera_yaw)
		sin_p := math.sin(camera_pitch)
		cos_p := math.cos(camera_pitch)

		forward_vec := Vec3{-sin_y * cos_p, -sin_p, -cos_y * cos_p}
		right_vec := Vec3{cos_y, 0.0, -sin_y}

		move_speed := f32(5.0)
		if movement.shift {
			move_speed = 20.0
		}
		dt_move := move_speed * delta_time

		if movement.forward {
			camera_pos[0] += forward_vec[0] * dt_move
			camera_pos[1] += forward_vec[1] * dt_move
			camera_pos[2] += forward_vec[2] * dt_move
		}
		if movement.backward {
			camera_pos[0] -= forward_vec[0] * dt_move
			camera_pos[1] -= forward_vec[1] * dt_move
			camera_pos[2] -= forward_vec[2] * dt_move
		}
		if movement.left {
			camera_pos[0] -= right_vec[0] * dt_move
			camera_pos[1] -= right_vec[1] * dt_move
			camera_pos[2] -= right_vec[2] * dt_move
		}
		if movement.right {
			camera_pos[0] += right_vec[0] * dt_move
			camera_pos[1] += right_vec[1] * dt_move
			camera_pos[2] += right_vec[2] * dt_move
		}
		if movement.up {
			camera_pos[1] += dt_move
		}
		if movement.down {
			camera_pos[1] -= dt_move
		}

		view_mat := linalg.MATRIX4F32_IDENTITY
		view_mat = linalg.matrix4_rotate_f32(-camera_yaw, {0, 1, 0}) * view_mat
		view_mat = linalg.matrix4_rotate_f32(-camera_pitch, {1, 0, 0}) * view_mat
		view_mat =
			view_mat *
			linalg.matrix4_translate_f32({-camera_pos[0], -camera_pos[1], -camera_pos[2]})

		// -- audio --
		if sdl.GetAudioStreamAvailable(stream) < cast(i32)wav_data_len {
			// sdl.PutAudioStreamData(stream, wav_data, cast(i32)wav_data_len)
			// my sister asked to comment this out because its loud and its late
		}

		// -- render --
		cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)
		swapchain_tex: ^sdl.GPUTexture
		ok = sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, window, &swapchain_tex, nil, nil)
		assert(ok)

		if swapchain_tex != nil {
			color_target := sdl.GPUColorTargetInfo {
				texture           = msaa_color_texture,
				load_op           = .CLEAR,
				clear_color       = {0.2, 0.4, 0.8, 1.0},
				store_op          = .RESOLVE,
				resolve_texture   = swapchain_tex,
				resolve_mip_level = 0,
				resolve_layer     = 0,
			}
			depth_target := sdl.GPUDepthStencilTargetInfo {
				texture          = msaa_depth_texture,
				clear_depth      = 1.0,
				load_op          = .CLEAR,
				store_op         = .STORE,
				stencil_load_op  = .DONT_CARE,
				stencil_store_op = .DONT_CARE,
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
				object_model := obj.local_model
				mvp := proj_mat * view_mat * object_model

				sdl.PushGPUVertexUniformData(cmd_buf, 0, &mvp, size_of(mvp))
				sdl.BindGPUFragmentSamplers(
					render_pass,
					0,
					&(sdl.GPUTextureSamplerBinding{texture = texture, sampler = sampler}),
					1,
				)
				sdl.DrawGPUIndexedPrimitives(
					render_pass,
					u32(obj.vertex_count),
					1,
					u32(obj.vertex_offset),
					0,
					0,
				)
			}

			sdl.EndGPURenderPass(render_pass)
		}

		ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
		assert(ok)
	}

	log.debug("Goodbye!")
}

load_shader :: proc(
	device: ^sdl.GPUDevice,
	code: []u8,
	stage: sdl.GPUShaderStage,
	num_uniform_buffers: u32,
	num_samplers: u32,
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
			num_samplers        = num_samplers,
		},
	)
}

load_mesh_data :: proc(model_path: string) -> (vertices: []VertexData, indices: []u16) {
	model_path_cstr := strings.clone_to_cstring(model_path)
	defer delete(model_path_cstr)

	options: cgltf.options
	data, result := cgltf.parse_file(options, model_path_cstr)
	assert(result == .success)
	result = cgltf.load_buffers(options, data, model_path_cstr)
	assert(result == .success)
	defer cgltf.free(data)

	mesh := data.scene.nodes[0].mesh
	primitive := mesh.primitives[0]

	pos_attr: ^cgltf.attribute = nil
	col_attr: ^cgltf.attribute = nil
	uv_attr: ^cgltf.attribute = nil
	for &attr in primitive.attributes {
		#partial switch attr.type {
		case cgltf.attribute_type.position:
			pos_attr = &attr
		case cgltf.attribute_type.color:
			col_attr = &attr
		case cgltf.attribute_type.texcoord:
			uv_attr = &attr
		}
	}
	assert(pos_attr != nil)

	pos_accessor := pos_attr.data
	col_accessor := col_attr != nil ? col_attr.data : nil
	uv_accessor := uv_attr.data
	idx_accessor := primitive.indices

	vertex_count := pos_accessor.count
	index_count := idx_accessor.count
	num_pos_components := cgltf.num_components(pos_accessor.type)
	num_col_components := col_accessor != nil ? cgltf.num_components(col_accessor.type) : 0
	num_uv_components := cgltf.num_components(uv_accessor.type)

	positions := make([]f32, vertex_count * num_pos_components)
	_ = cgltf.accessor_unpack_floats(
		pos_accessor,
		raw_data(positions),
		uint(vertex_count * num_pos_components),
	)

	colors: []f32
	if col_accessor != nil {
		colors = make([]f32, vertex_count * num_col_components)
		_ = cgltf.accessor_unpack_floats(
			col_accessor,
			raw_data(colors),
			uint(vertex_count * num_col_components),
		)
	}

	uvs := make([]f32, vertex_count * num_uv_components)
	_ = cgltf.accessor_unpack_floats(
		uv_accessor,
		raw_data(uvs),
		uint(vertex_count * num_uv_components),
	)

	vertices = make([]VertexData, vertex_count)
	for i := 0; i < int(vertex_count); i += 1 {
		pos_idx := i * int(num_pos_components)
		col_idx := i * int(num_col_components)
		uv_idx := i * int(num_uv_components)

		color: sdl.FColor
		if col_accessor != nil {
			color = sdl.FColor{colors[col_idx + 0], colors[col_idx + 1], colors[col_idx + 2], 1.0}
		} else {
			color = sdl.FColor{1.0, 1.0, 1.0, 1.0}
		}

		vertices[i] = VertexData {
			pos   = Vec3{positions[pos_idx + 0], positions[pos_idx + 1], positions[pos_idx + 2]},
			color = color,
			uv    = [2]f32{uvs[uv_idx + 0], uvs[uv_idx + 1]},
		}
	}

	indices = make([]u16, index_count)
	for i := 0; i < int(index_count); i += 1 {
		indices[i] = u16(cgltf.accessor_read_index(idx_accessor, uint(i)))
	}

	return vertices, indices
}

package main

import "base:runtime"
import clay "clay-odin"
import "core:flags"
import "core:fmt"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:reflect"
import "core:strings"
import "vendor:box2d"
import ma "vendor:miniaudio"
import sdl "vendor:sdl3"
import stbi "vendor:stb/image"

default_context: runtime.Context

ma_engine: ma.engine

render_game: bool = true
render_ui: bool = false

show_collision_debug: bool = false

water_height: f32 = 18

world_id: box2d.WorldId

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

	when ODIN_DEBUG {
		os.set_env("MTL_DEBUG_LAYER", "1")

		tracking_allocator: mem.Tracking_Allocator
		mem.tracking_allocator_init(&tracking_allocator, context.allocator)
		context.allocator = mem.tracking_allocator(&tracking_allocator)
		defer {
			leaked := len(tracking_allocator.allocation_map) > 0
			if leaked {
				fmt.eprint("\n--== Memory Leaks ==--\n")
				fmt.eprintf("Total Leaks: %v\n", len(tracking_allocator.allocation_map))
				for _, leak in tracking_allocator.allocation_map {
					fmt.eprintf("Leak: %v bytes @%v\n", leak.size, leak.location)
				}
			}

			bad_frees := len(tracking_allocator.bad_free_array) > 0
			if bad_frees {
				fmt.eprint("\n--== Bad Frees ==--\n")
				for bad_free in tracking_allocator.bad_free_array {
					fmt.eprintf("Bad Free: %p @%v\n", bad_free.memory, bad_free.location)
				}
			}

			if !leaked && !bad_frees {
				fmt.println("No leaks or bad frees detected!")
			}
		}
	}

	EntryPoints :: enum {
		Game,
		Map,
	}

	Options :: struct {
		entry_point: EntryPoints `args:"pos=0" usage:"Entry point."`,
	}

	opt: Options
	flags.parse_or_exit(&opt, os.args, .Odin)

	switch opt.entry_point {
	case .Game:
		break
	case .Map:
		test_bsp_data()
		return
	}

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

	ok := sdl.SetAppMetadata("arctic char*", "0.1.0", "sh.fogo.arctic-char")
	assert(ok)

	ok = sdl.Init({.VIDEO, .AUDIO})
	assert(ok)
	defer sdl.Quit()

	// -- end initial setup --

	// -- audio setup --

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

	window := sdl.CreateWindow("arctic char*", 1024, 512, {.RESIZABLE})
	assert(window != nil)

	gpu := sdl.CreateGPUDevice(shader_format, gpu_debug, nil)
	assert(gpu != nil)

	ok = sdl.ClaimWindowForGPUDevice(gpu, window)
	assert(ok)

	win_size: [2]i32
	ok = sdl.GetWindowSize(window, &win_size.x, &win_size.y)
	assert(ok)

	msaa_color_texture, msaa_depth_texture := create_msaa_textures(
		gpu,
		window,
		win_size.x,
		win_size.y,
	)
	defer sdl.ReleaseGPUTexture(gpu, msaa_color_texture)
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
		num_uniform_buffers = 1,
		num_samplers = 1,
	)

	img_size: [2]i32
	pixels := stbi.load("./assets/atlas.png", &img_size.x, &img_size.y, nil, 4)
	assert(pixels != nil)
	pixels_byte_size := int(img_size.x * img_size.y * 4)

	log.debugf("Pixels byte size: %d", pixels_byte_size)

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
	sdl.SetGPUTextureName(gpu, texture, "Test Texture")

	mesh_vertex_buffer_description := sdl.GPUVertexBufferDescription {
		slot               = 0,
		pitch              = size_of(f32) * (3 + 4 + 3), // float3 pos + float4 color + float3 uvw
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
	mesh_vertex_attribute_uvw := sdl.GPUVertexAttribute {
		location    = 2,
		buffer_slot = 0,
		format      = .FLOAT3,
		offset      = size_of(f32) * (3 + 4), // After float3 position + float4 color
	}
	mesh_vertex_attributes := [3]sdl.GPUVertexAttribute {
		mesh_vertex_attribute_position,
		mesh_vertex_attribute_color,
		mesh_vertex_attribute_uvw,
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

		if model_name == "Map" {
			continue
		}

		model_name_lower := strings.to_lower(model_name)
		model_path := fmt.tprintf("./assets/%s.glb", model_name_lower)

		vertex_data, index_data := load_mesh_data(model_path)

		if model_name == "Collision" {
			world_def := box2d.DefaultWorldDef()
			world_def.gravity = box2d.Vec2{0, -10}
			world_id = box2d.CreateWorld(world_def)
			map_body := load_box2d_geometry(world_id, vertex_data, index_data)
		}

		for j in 0 ..< len(index_data) {
			index_data[j] += u16(combined_vertex_count)
		}

		vertex_datas[i] = vertex_data
		index_datas[i] = index_data

		model_enum, ok := reflect.enum_from_name(Model, model_name)
		assert(ok)

		log.debugf(
			"Loaded Model '%s', index_offset=%d, index_count=%d, vertex_count=%d",
			model_name,
			combined_index_count,
			len(index_data),
			len(vertex_data),
		)

		model_info_lookup[model_enum] = ModelInfo {
			index_offset = combined_index_count,
			index_count  = len(index_data),
		}

		combined_vertex_count += len(vertex_data)
		combined_index_count += len(index_data)
	}

	{
		bsp_data, map_ok := load_bsp("./assets/maps/test.bsp")
		assert(map_ok)

		vertices, indices := bsp_to_model(&bsp_data)

		for i in 0 ..< len(indices) {
			indices[i] += u16(combined_vertex_count)
		}

		map_model_enum := Model.Map
		model_info_lookup[map_model_enum] = ModelInfo {
			index_offset = combined_index_count,
			index_count  = len(indices),
		}

		log.debugf(
			"Loaded Map Model, index_offset=%d, index_count=%d, vertex_count=%d",
			combined_index_count,
			len(indices),
			len(vertices),
		)

		combined_vertex_count += len(vertices)
		combined_index_count += len(indices)

		vertex_datas[Model.Map] = vertices
		index_datas[Model.Map] = indices

		free_bsp_data(&bsp_data)
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

	combined_vertex_data_size := combined_vertex_count * size_of(VertexData)
	combined_index_data_size := combined_index_count * size_of(u16)
	texture_size := pixels_byte_size

	log.debugf(
		"Memory usage: vertex data=%.2f kb, index data=%.2f kb, texture=%.2f kb",
		f32(combined_vertex_data_size) / 1024.0,
		f32(combined_index_data_size) / 1024.0,
		f32(texture_size) / 1024.0,
	)

	total_size_in_bytes := combined_vertex_data_size + combined_index_data_size + texture_size

	transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = u32(total_size_in_bytes), props = 0},
	)

	transfer_buffer_ptr := transmute([^]byte)sdl.MapGPUTransferBuffer(gpu, transfer_buffer, false)
	transfer_buffer_offset := 0

	log.debugf("Copying vertex data to transfer buffer")
	mem.copy(
		transfer_buffer_ptr[transfer_buffer_offset:],
		raw_data(combined_vertex_data),
		len(combined_vertex_data) * size_of(VertexData),
	)
	transfer_buffer_offset += len(combined_vertex_data) * size_of(VertexData)

	log.debugf("Copying index data to transfer buffer")
	mem.copy(
		transfer_buffer_ptr[transfer_buffer_offset:],
		raw_data(combined_index_data),
		len(combined_index_data) * size_of(u16),
	)
	transfer_buffer_offset += len(combined_index_data) * size_of(u16)

	texture_offset := transfer_buffer_offset
	if texture_offset % 4 != 0 {
		padding := 4 - (texture_offset % 4)
		texture_offset += padding
	}

	mem.copy(transfer_buffer_ptr[texture_offset:], pixels, pixels_byte_size)

	sdl.UnmapGPUTransferBuffer(gpu, transfer_buffer)

	for vertex_data in vertex_datas {
		delete(vertex_data)
	}

	for index_data in index_datas {
		delete(index_data)
	}

	vertex_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.VERTEX}, size = u32(combined_vertex_data_size), props = 0},
	)
	sdl.SetGPUBufferName(gpu, vertex_buffer, "Main Vertex Buffer")

	index_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.INDEX}, size = u32(combined_index_data_size), props = 0},
	)
	sdl.SetGPUBufferName(gpu, index_buffer, "Main Index Buffer")

	copy_command_buffer := sdl.AcquireGPUCommandBuffer(gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_command_buffer)

	buffer_offset: u32 = 0

	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = buffer_offset},
		{
			buffer = vertex_buffer,
			offset = 0,
			size = u32(combined_vertex_count * size_of(VertexData)),
		},
		false,
	)

	buffer_offset += u32(combined_vertex_count * size_of(VertexData))

	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = buffer_offset},
		{buffer = index_buffer, offset = 0, size = u32(combined_index_count * size_of(u16))},
		false,
	)

	buffer_offset = u32(texture_offset) // Use the same offset we calculated earlier

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

	last_ticks := sdl.GetTicks()
	total_time: f32 = 0.0

	camera := Camera {
		mode              = .Player,
		perspective       = .Perspective,
		mouse_sensitivity = 0.003,
	}

	proj_mat := camera_get_projection(&camera, win_size.x, win_size.y)

	movement := Movement{}

	log.debug("Ready for main loop")

	clayErrorhandler :: proc "c" (errorData: clay.ErrorData) {
		context = default_context

		message := string(errorData.errorText.chars[:errorData.errorText.length])

		log.errorf("Clay error: {}", message)
	}

	minMemorySize: u32 = clay.MinMemorySize()
	clay_memory := make([^]u8, minMemorySize)
	defer mem.free(clay_memory)
	arena: clay.Arena = clay.CreateArenaWithCapacityAndMemory(minMemorySize, clay_memory)
	clay.Initialize(
		arena,
		clay.Dimensions{width = f32(win_size.x), height = f32(win_size.y)},
		{handler = clayErrorhandler},
	)

	clay.SetMeasureTextFunction(claySdlGpuRenderMeasureText, nil)

	if ODIN_DEBUG {
		clay.SetDebugModeEnabled(true)
		// todo make keyboard shortcut to toggle debug mode
	}

	ui_pipeline := claySdlGpuRenderInitialize(gpu, window)

	log.debug("Clay initialized")

	entity_create_test_entities(Vec3{0, 0, -10})

	fragment_uniforms: struct {
		atlas_width:  f32,
		atlas_height: f32,
		atlas_lookup: [len(atlas_textures)][4]f32,
	}

	fragment_uniforms.atlas_width = f32(ATLAS_WIDTH)
	fragment_uniforms.atlas_height = f32(ATLAS_HEIGHT)

	for texture, i in atlas_textures {
		fragment_uniforms.atlas_lookup[i] = {
			texture.rect.width,
			texture.rect.height,
			texture.rect.x,
			texture.rect.y,
		}
	}

	time_accumulator: f32 = 0.0

	main_loop: for {
		new_ticks := sdl.GetTicks()
		delta_time := f32(new_ticks - last_ticks) / 1000
		last_ticks = new_ticks
		total_time += delta_time

		PHYSICS_STEP :: 1 / 60.0

		time_accumulator += delta_time
		for time_accumulator >= PHYSICS_STEP {
			box2d.World_Step(world_id, PHYSICS_STEP, 4)

			contact_events := box2d.World_GetContactEvents(world_id)

			for i in 0 ..< contact_events.beginCount {
				contact_event := contact_events.beginEvents[i]

				if contact_event.shapeIdA == player.shape ||
				   contact_event.shapeIdB == player.shape {
					other_shape_id: box2d.ShapeId
					if contact_event.shapeIdA == player.shape {
						other_shape_id = contact_event.shapeIdB
					} else {
						other_shape_id = contact_event.shapeIdA
					}

					for entity_idx := 0; entity_idx < len(entities); entity_idx += 1 {
						if entities[entity_idx].has_body &&
						   entities[entity_idx].shape == other_shape_id {
							entity_player_collide(&entities[entity_idx])
							break
						}
					}
				}
			}

			time_accumulator -= PHYSICS_STEP
		}

		entity_update_all()

		// -- events --
		ev: sdl.Event
		for sdl.PollEvent(&ev) {
			#partial switch ev.type {
			case .QUIT:
				break main_loop

			case .WINDOW_RESIZED:
				ok = sdl.GetWindowSize(window, &win_size.x, &win_size.y)
				assert(ok)

				log.debugf("Window resize: {}x{}", win_size.x, win_size.y)

				sdl.ReleaseGPUTexture(gpu, msaa_color_texture)
				sdl.ReleaseGPUTexture(gpu, msaa_depth_texture)

				msaa_color_texture, msaa_depth_texture = create_msaa_textures(
					gpu,
					window,
					win_size.x,
					win_size.y,
				)

				proj_mat = camera_get_projection(&camera, win_size.x, win_size.y)

			case .MOUSE_BUTTON_DOWN:
				if ev.button.button == 1 {
					camera_handle_mouse_button(&camera, 1, true, window)
				}

			case .KEY_DOWN:
				if ev.key.scancode == .Q {
					break main_loop
				} else if ev.key.scancode == .ESCAPE {
					camera_unlock(&camera, window)
				} else if ev.key.scancode == .C {
					if camera.mode == .Noclip {
						camera.mode = .Player
					} else {
						camera.mode = .Noclip
						camera_lock(&camera, window)
					}
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
				} else if ev.key.scancode == .P {
					if camera.perspective == .Perspective {
						camera.perspective = .Orthographic
					} else {
						camera.perspective = .Perspective
					}
					proj_mat = camera_get_projection(&camera, win_size.x, win_size.y)
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
				camera_handle_mouse_motion(&camera, f32(ev.motion.xrel), f32(ev.motion.yrel))
			}
		}

		view_mat: matrix[4, 4]f32
		view_mat = camera_update(&camera, delta_time, movement)

		// -- audio --
		if sdl.GetAudioStreamAvailable(stream) < cast(i32)wav_data_len {
			// sdl.PutAudioStreamData(stream, wav_data, cast(i32)wav_data_len)
			// my sister asked to comment this out because its loud and its late
		}

		// -- render --
		cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)
		if ODIN_DEBUG {
			sdl.InsertGPUDebugLabel(cmd_buf, "Main Loop Render Pass")
		}
		swapchain_tex: ^sdl.GPUTexture
		ok = sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, window, &swapchain_tex, nil, nil)
		assert(ok)

		if swapchain_tex != nil {
			if render_game {
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

				sdl.BindGPUFragmentSamplers(
					render_pass,
					0,
					&(sdl.GPUTextureSamplerBinding{texture = texture, sampler = sampler}),
					1,
				)

				sdl.PushGPUFragmentUniformData(
					cmd_buf,
					0,
					&fragment_uniforms,
					size_of(fragment_uniforms),
				)

				for entity, i in entities {
					object_model := entity.local_model

					vertex_uniforms := struct {
						mv:            matrix[4, 4]f32,
						proj:          matrix[4, 4]f32,
						viewport_size: [2]f32,
					} {
						mv            = view_mat * object_model,
						proj          = proj_mat,
						viewport_size = [2]f32{f32(win_size.x), f32(win_size.y)},
					}
					sdl.PushGPUVertexUniformData(
						cmd_buf,
						0,
						&vertex_uniforms,
						size_of(vertex_uniforms),
					)

					sdl.DrawGPUIndexedPrimitives(
						render_pass,
						u32(entity.model_info.index_count),
						1,
						u32(entity.model_info.index_offset),
						0,
						0,
					)
				}

				sdl.EndGPURenderPass(render_pass)
			}

			if render_ui {
				ui_color_target := sdl.GPUColorTargetInfo {
					texture  = swapchain_tex,
					store_op = .STORE,
					load_op  = .LOAD,
				}

				ui_render_pass := sdl.BeginGPURenderPass(cmd_buf, &ui_color_target, 1, nil)
				sdl.BindGPUGraphicsPipeline(ui_render_pass, ui_pipeline)

				ui_proj := matrix4_orthographic_f32(0, f32(win_size.x), f32(win_size.y), 0, -1, 1)

				clay.SetLayoutDimensions({f32(win_size.x), f32(win_size.y)})

				clay.BeginLayout()

				COLOR_RED :: clay.Color{255, 0, 0, 255}
				COLOR_GREEN :: clay.Color{0, 255, 0, 255}
				COLOR_BLUE :: clay.Color{0, 0, 255, 255}
				COLOR_YELLOW :: clay.Color{255, 255, 0, 255}
				COLOR_PURPLE :: clay.Color{255, 0, 255, 255}
				COLOR_CYAN :: clay.Color{0, 255, 255, 255}

				if clay.UI()(
				{
					id = clay.ID("OuterContainer"),
					layout = {
						layoutDirection = .LeftToRight,
						sizing = {clay.SizingFixed(100), clay.SizingGrow({})},
						padding = {10, 10, 10, 10},
					},
					backgroundColor = COLOR_CYAN,
				},
				) {
					if clay.UI()(
					{
						id = clay.ID("InnerContainer"),
						layout = {
							layoutDirection = .TopToBottom,
							sizing = {clay.SizingGrow({}), clay.SizingGrow({})},
						},
						backgroundColor = COLOR_RED,
					},
					) {}
				}
				clay_commands := clay.EndLayout()

				claySdlGpuRender(&clay_commands, gpu, cmd_buf, ui_render_pass, ui_proj)

				sdl.EndGPURenderPass(ui_render_pass)
			}
		}

		ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
	}

	delete(model_info_lookup)

	log.debug("Goodbye!")
}

@(export)
NvOptimusEnablement: u32 = 1

@(export)
AmdPowerXpressRequestHighPerformance: i32 = 1

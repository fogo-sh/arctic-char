package game

import "base:runtime"
import "core:mem"
import "core:log"
import sdl "vendor:sdl3"

Renderer :: struct {
	allocator: runtime.Allocator,

	gpu:    ^sdl.GPUDevice,
	window: ^sdl.Window,

	pipeline: ^sdl.GPUGraphicsPipeline,

	msaa_color_texture: ^sdl.GPUTexture,
	depth_texture:      ^sdl.GPUTexture,

	meshes: [dynamic]GpuMesh,

	sample_count: sdl.GPUSampleCount,
}

GpuMesh :: struct {
	vertex_buffer: ^sdl.GPUBuffer,
	index_buffer:  ^sdl.GPUBuffer,
	index_count:   int,
}

MeshHandle :: distinct int

RenderItem :: struct {
	mesh: MeshHandle,
	mvp: matrix[4, 4]f32,
}

// Creates the renderer-owned GPU objects: render targets, pipeline, and mesh buffers.
// The caller still owns the SDL window/device and the CPU mesh memory.
renderer_create :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	width, height: i32,
	meshes: []CpuMesh,
	allocator := context.allocator,
) -> Renderer {
	renderer := Renderer{
		allocator = allocator,
		gpu = gpu,
		window = window,
		sample_count = renderer_choose_sample_count(gpu, window),
	}
	log.debugf("Renderer MSAA sample count: %v", renderer.sample_count)
	renderer_create_render_targets(&renderer, width, height)
	renderer.pipeline = renderer_create_pipeline(gpu, window, renderer.sample_count)
	renderer.meshes = make([dynamic]GpuMesh, 0, len(meshes), allocator)
	for &mesh in meshes {
		append(&renderer.meshes, renderer_upload_mesh(&renderer, &mesh))
	}
	return renderer
}

renderer_destroy :: proc(renderer: ^Renderer) {
	for &mesh in renderer.meshes {
		renderer_destroy_mesh(renderer, &mesh)
	}
	delete(renderer.meshes)
	if renderer.depth_texture != nil do sdl.ReleaseGPUTexture(renderer.gpu, renderer.depth_texture)
	if renderer.msaa_color_texture != nil do sdl.ReleaseGPUTexture(renderer.gpu, renderer.msaa_color_texture)
	if renderer.pipeline != nil do sdl.ReleaseGPUGraphicsPipeline(renderer.gpu, renderer.pipeline)
	renderer^ = {}
}

renderer_resize :: proc(renderer: ^Renderer, width, height: i32) {
	if renderer.depth_texture != nil do sdl.ReleaseGPUTexture(renderer.gpu, renderer.depth_texture)
	if renderer.msaa_color_texture != nil do sdl.ReleaseGPUTexture(renderer.gpu, renderer.msaa_color_texture)
	renderer_create_render_targets(renderer, width, height)
}

renderer_choose_sample_count :: proc(gpu: ^sdl.GPUDevice, window: ^sdl.Window) -> sdl.GPUSampleCount {
	color_format := sdl.GetGPUSwapchainTextureFormat(gpu, window)
	color_supports_4x := sdl.GPUTextureSupportsSampleCount(gpu, color_format, ._4)
	depth_supports_4x := sdl.GPUTextureSupportsSampleCount(gpu, .D32_FLOAT, ._4)
	if color_supports_4x && depth_supports_4x {
		return ._4
	}
	return ._1
}

// MSAA renders into a multisampled color texture, then resolves into the swapchain.
// Depth must use the same sample count as the color target.
renderer_create_render_targets :: proc(renderer: ^Renderer, width, height: i32) {
	if renderer.sample_count != ._1 {
		renderer.msaa_color_texture = sdl.CreateGPUTexture(
			renderer.gpu,
			{
				type = .D2,
				format = sdl.GetGPUSwapchainTextureFormat(renderer.gpu, renderer.window),
				usage = {.COLOR_TARGET},
				width = u32(width),
				height = u32(height),
				layer_count_or_depth = 1,
				num_levels = 1,
				sample_count = renderer.sample_count,
			},
		)
		assert(renderer.msaa_color_texture != nil)
	} else {
		renderer.msaa_color_texture = nil
	}

	renderer.depth_texture = sdl.CreateGPUTexture(
		renderer.gpu,
		{
			type = .D2,
			format = .D32_FLOAT,
			usage = {.DEPTH_STENCIL_TARGET},
			width = u32(width),
			height = u32(height),
			layer_count_or_depth = 1,
			num_levels = 1,
			sample_count = renderer.sample_count,
		},
	)
	assert(renderer.depth_texture != nil)
}

renderer_create_pipeline :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	sample_count: sdl.GPUSampleCount,
) -> ^sdl.GPUGraphicsPipeline {
	vert_shader := load_shader(
		gpu,
		vert_shader_code,
		.VERTEX,
		num_uniform_buffers = 1,
		entrypoint_name = "VertexMain",
	)
	frag_shader := load_shader(
		gpu,
		frag_shader_code,
		.FRAGMENT,
		num_uniform_buffers = 0,
		entrypoint_name = "FragmentMain",
	)
	assert(vert_shader != nil)
	assert(frag_shader != nil)
	defer sdl.ReleaseGPUShader(gpu, vert_shader)
	defer sdl.ReleaseGPUShader(gpu, frag_shader)

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
		depth_stencil_format      = .D32_FLOAT,
		has_depth_stencil_target  = true,
	}
	depth_stencil_state := sdl.GPUDepthStencilState {
		compare_op = .LESS,
		enable_depth_test = true,
		enable_depth_write = true,
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
			multisample_state = {
				sample_count = sample_count,
			},
			depth_stencil_state = depth_stencil_state,
			target_info = target_info,
		},
	)
	assert(pipeline != nil)
	return pipeline
}

renderer_destroy_mesh :: proc(renderer: ^Renderer, mesh: ^GpuMesh) {
	if mesh.index_buffer != nil do sdl.ReleaseGPUBuffer(renderer.gpu, mesh.index_buffer)
	if mesh.vertex_buffer != nil do sdl.ReleaseGPUBuffer(renderer.gpu, mesh.vertex_buffer)
	mesh^ = {}
}

// Uploads one immutable mesh by staging vertex/index data through a transfer buffer.
renderer_upload_mesh :: proc(renderer: ^Renderer, mesh: ^CpuMesh) -> GpuMesh {
	assert(len(mesh.vertices) > 0)
	assert(len(mesh.indices) > 0)
	vertex_data_size := len(mesh.vertices) * size_of(VertexData)
	index_data_size := len(mesh.indices) * size_of(u32)
	total_upload_size := vertex_data_size + index_data_size
	gpu_mesh: GpuMesh

	transfer_buffer := sdl.CreateGPUTransferBuffer(
		renderer.gpu,
		{usage = .UPLOAD, size = u32(total_upload_size), props = 0},
	)
	assert(transfer_buffer != nil)
	defer sdl.ReleaseGPUTransferBuffer(renderer.gpu, transfer_buffer)

	transfer_ptr := transmute([^]byte)sdl.MapGPUTransferBuffer(renderer.gpu, transfer_buffer, false)
	mem.copy(transfer_ptr[:], raw_data(mesh.vertices), vertex_data_size)
	mem.copy(transfer_ptr[vertex_data_size:], raw_data(mesh.indices), index_data_size)
	sdl.UnmapGPUTransferBuffer(renderer.gpu, transfer_buffer)

	gpu_mesh.vertex_buffer = sdl.CreateGPUBuffer(
		renderer.gpu,
		{usage = {.VERTEX}, size = u32(vertex_data_size), props = 0},
	)
	assert(gpu_mesh.vertex_buffer != nil)

	gpu_mesh.index_buffer = sdl.CreateGPUBuffer(
		renderer.gpu,
		{usage = {.INDEX}, size = u32(index_data_size), props = 0},
	)
	assert(gpu_mesh.index_buffer != nil)
	gpu_mesh.index_count = len(mesh.indices)

	copy_cmd := sdl.AcquireGPUCommandBuffer(renderer.gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_cmd)
	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{buffer = gpu_mesh.vertex_buffer, offset = 0, size = u32(vertex_data_size)},
		false,
	)
	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = u32(vertex_data_size)},
		{buffer = gpu_mesh.index_buffer, offset = 0, size = u32(index_data_size)},
		false,
	)
	sdl.EndGPUCopyPass(copy_pass)
	ok := sdl.SubmitGPUCommandBuffer(copy_cmd)
	assert(ok)
	return gpu_mesh
}

renderer_draw :: proc(
	renderer: ^Renderer,
	cmd_buf: ^sdl.GPUCommandBuffer,
	swapchain_tex: ^sdl.GPUTexture,
	items: []RenderItem,
) {
	if swapchain_tex == nil do return

	color_target := renderer_color_target(renderer, swapchain_tex)
	depth_target := sdl.GPUDepthStencilTargetInfo {
		texture          = renderer.depth_texture,
		clear_depth      = 1.0,
		load_op          = .CLEAR,
		store_op         = .DONT_CARE,
		stencil_load_op  = .DONT_CARE,
		stencil_store_op = .DONT_CARE,
	}

	render_pass := sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, &depth_target)
	sdl.BindGPUGraphicsPipeline(render_pass, renderer.pipeline)

	for item in items {
		mesh := renderer_mesh(renderer, item.mesh)
		vertex_buffer_binding := sdl.GPUBufferBinding{buffer = mesh.vertex_buffer, offset = 0}
		index_buffer_binding := sdl.GPUBufferBinding{buffer = mesh.index_buffer, offset = 0}
		sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)
		sdl.BindGPUIndexBuffer(render_pass, index_buffer_binding, ._32BIT)

		vertex_uniforms := struct {
			mvp: matrix[4, 4]f32,
		}{mvp = item.mvp}
		sdl.PushGPUVertexUniformData(cmd_buf, 0, &vertex_uniforms, size_of(vertex_uniforms))
		sdl.DrawGPUIndexedPrimitives(render_pass, u32(mesh.index_count), 1, 0, 0, 0)
	}
	sdl.EndGPURenderPass(render_pass)
}

renderer_mesh :: proc(renderer: ^Renderer, handle: MeshHandle) -> ^GpuMesh {
	index := int(handle)
	assert(0 <= index && index < len(renderer.meshes))
	return &renderer.meshes[index]
}

renderer_color_target :: proc(
	renderer: ^Renderer,
	swapchain_tex: ^sdl.GPUTexture,
) -> sdl.GPUColorTargetInfo {
	color_target := sdl.GPUColorTargetInfo {
		load_op     = .CLEAR,
		clear_color = {0.04, 0.05, 0.07, 1.0},
	}

	if renderer.sample_count == ._1 {
		color_target.texture = swapchain_tex
		color_target.store_op = .STORE
		return color_target
	}

	color_target.texture = renderer.msaa_color_texture
	color_target.store_op = .RESOLVE
	color_target.resolve_texture = swapchain_tex
	return color_target
}

package engine

import "base:runtime"
import "core:mem"
import "core:log"
import sdl "vendor:sdl3"

Renderer :: struct {
	allocator: runtime.Allocator,

	gpu:    ^sdl.GPUDevice,
	window: ^sdl.Window,

	pipeline:     ^sdl.GPUGraphicsPipeline,
	line_pipeline: ^sdl.GPUGraphicsPipeline,
	sky_pipeline: ^sdl.GPUGraphicsPipeline,
	ui_pipeline: ^sdl.GPUGraphicsPipeline,
	text:        ^TextRenderer,
	debug_line_vertex_buffer: ^sdl.GPUBuffer,
	debug_line_transfer_buffer: ^sdl.GPUTransferBuffer,
	debug_line_vertices: [dynamic]VertexData,

	msaa_color_texture: ^sdl.GPUTexture,
	depth_texture:      ^sdl.GPUTexture,

	meshes: [dynamic]GpuMesh,

	sample_count: sdl.GPUSampleCount,
	stats: RendererStats,
}

GpuMesh :: struct {
	vertex_buffer: ^sdl.GPUBuffer,
	index_buffer:  ^sdl.GPUBuffer,
	index_count:   int,
}

MeshHandle :: distinct int

RenderItem :: struct {
	mesh:  MeshHandle,
	model: matrix[4, 4]f32,
}

DebugLine :: struct {
	from:  Vec3,
	to:    Vec3,
	color: Color,
}

DEBUG_LINE_CAPACITY :: 32768

RenderPassGlobals :: struct {
	view:        matrix[4, 4]f32,
	proj:        matrix[4, 4]f32,
	environment: RenderEnvironment,
}

RenderEnvironment :: struct {
	fog_color:         [4]f32,
	sky_top_color:     [4]f32,
	sky_horizon_color: [4]f32,
	fog_distances:     [4]f32,
}

WorldVertexUniforms :: struct {
	mvp:        matrix[4, 4]f32,
	model_view: matrix[4, 4]f32,
}

FragmentUniforms :: struct {
	fog_color:         [4]f32,
	sky_top_color:     [4]f32,
	sky_horizon_color: [4]f32,
	fog_distances:     [4]f32,
}

RendererStats :: struct {
	draw_count:     int,
	triangle_count: int,
	mesh_count:     int,
	pipeline_count: int,
}

RendererUpload :: struct {
	renderer:         ^Renderer,
	cmd_buf:          ^sdl.GPUCommandBuffer,
	copy_pass:        ^sdl.GPUCopyPass,
	transfer_buffers: [dynamic]^sdl.GPUTransferBuffer,
}

// Creates the renderer-owned GPU objects: render targets, pipeline, and mesh buffers.
// The caller still owns the SDL window/device and the CPU mesh memory.
renderer_create :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	width, height: i32,
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
	renderer.line_pipeline = renderer_create_line_pipeline(gpu, window, renderer.sample_count)
	renderer.sky_pipeline = renderer_create_sky_pipeline(gpu, window, renderer.sample_count)
	renderer.ui_pipeline = renderer_create_ui_pipeline(gpu, window, renderer.sample_count)
	renderer.text = renderer_create_text(gpu, window, renderer.sample_count)
	renderer.meshes = make([dynamic]GpuMesh, 0, 16, allocator)
	renderer.debug_line_vertices = make([dynamic]VertexData, 0, DEBUG_LINE_CAPACITY * 2, allocator)
	renderer.debug_line_vertex_buffer = sdl.CreateGPUBuffer(gpu, {usage = {.VERTEX}, size = u32(DEBUG_LINE_CAPACITY * 2 * size_of(VertexData))})
	renderer.debug_line_transfer_buffer = sdl.CreateGPUTransferBuffer(gpu, {usage = .UPLOAD, size = u32(DEBUG_LINE_CAPACITY * 2 * size_of(VertexData))})
	assert(renderer.debug_line_vertex_buffer != nil)
	assert(renderer.debug_line_transfer_buffer != nil)
	renderer_update_static_stats(&renderer)
	return renderer
}

renderer_destroy :: proc(renderer: ^Renderer) {
	for &mesh in renderer.meshes {
		renderer_destroy_mesh(renderer, &mesh)
	}
	delete(renderer.debug_line_vertices)
	delete(renderer.meshes)
	if renderer.debug_line_transfer_buffer != nil do sdl.ReleaseGPUTransferBuffer(renderer.gpu, renderer.debug_line_transfer_buffer)
	if renderer.debug_line_vertex_buffer != nil do sdl.ReleaseGPUBuffer(renderer.gpu, renderer.debug_line_vertex_buffer)
	if renderer.depth_texture != nil do sdl.ReleaseGPUTexture(renderer.gpu, renderer.depth_texture)
	if renderer.msaa_color_texture != nil do sdl.ReleaseGPUTexture(renderer.gpu, renderer.msaa_color_texture)
	if renderer.pipeline != nil do sdl.ReleaseGPUGraphicsPipeline(renderer.gpu, renderer.pipeline)
	if renderer.line_pipeline != nil do sdl.ReleaseGPUGraphicsPipeline(renderer.gpu, renderer.line_pipeline)
	if renderer.sky_pipeline != nil do sdl.ReleaseGPUGraphicsPipeline(renderer.gpu, renderer.sky_pipeline)
	if renderer.ui_pipeline != nil do sdl.ReleaseGPUGraphicsPipeline(renderer.gpu, renderer.ui_pipeline)
	renderer_destroy_text(renderer.gpu, renderer.text)
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
		num_uniform_buffers = 1,
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
	vertex_attributes := [4]sdl.GPUVertexAttribute {
		{location = 0, buffer_slot = 0, format = .FLOAT3, offset = 0},
		{location = 1, buffer_slot = 0, format = .FLOAT3, offset = u32(offset_of(VertexData, normal))},
		{location = 2, buffer_slot = 0, format = .FLOAT2, offset = u32(offset_of(VertexData, uv))},
		{location = 3, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(VertexData, color))},
	}
	color_target_description := sdl.GPUColorTargetDescription {
		format = sdl.GetGPUSwapchainTextureFormat(gpu, window),
		blend_state = sdl.GPUColorTargetBlendState{},
	}
	target_info := sdl.GPUGraphicsPipelineTargetInfo {
		num_color_targets         = 1,
		color_target_descriptions = &color_target_description,
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
				enable_depth_clip = true,
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

renderer_create_line_pipeline :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	sample_count: sdl.GPUSampleCount,
) -> ^sdl.GPUGraphicsPipeline {
	vert_shader := load_shader(gpu, vert_shader_code, .VERTEX, num_uniform_buffers = 1, entrypoint_name = "VertexMain")
	frag_shader := load_shader(gpu, frag_shader_code, .FRAGMENT, num_uniform_buffers = 1, entrypoint_name = "FragmentMain")
	assert(vert_shader != nil)
	assert(frag_shader != nil)
	defer sdl.ReleaseGPUShader(gpu, vert_shader)
	defer sdl.ReleaseGPUShader(gpu, frag_shader)

	vertex_buffer_description := sdl.GPUVertexBufferDescription{slot = 0, pitch = size_of(VertexData), input_rate = .VERTEX}
	vertex_attributes := [4]sdl.GPUVertexAttribute{
		{location = 0, buffer_slot = 0, format = .FLOAT3, offset = 0},
		{location = 1, buffer_slot = 0, format = .FLOAT3, offset = u32(offset_of(VertexData, normal))},
		{location = 2, buffer_slot = 0, format = .FLOAT2, offset = u32(offset_of(VertexData, uv))},
		{location = 3, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(VertexData, color))},
	}
	color_target_description := sdl.GPUColorTargetDescription{format = sdl.GetGPUSwapchainTextureFormat(gpu, window)}
	target_info := sdl.GPUGraphicsPipelineTargetInfo{
		num_color_targets = 1,
		color_target_descriptions = &color_target_description,
		depth_stencil_format = .D32_FLOAT,
		has_depth_stencil_target = true,
	}
	pipeline := sdl.CreateGPUGraphicsPipeline(gpu, {
		vertex_shader = vert_shader,
		fragment_shader = frag_shader,
		vertex_input_state = {
			num_vertex_buffers = 1,
			vertex_buffer_descriptions = &vertex_buffer_description,
			num_vertex_attributes = len(vertex_attributes),
			vertex_attributes = &vertex_attributes[0],
		},
		primitive_type = .LINELIST,
		rasterizer_state = {front_face = .COUNTER_CLOCKWISE, enable_depth_clip = true},
		multisample_state = {sample_count = sample_count},
		depth_stencil_state = {compare_op = .LESS_OR_EQUAL, enable_depth_test = true, enable_depth_write = false},
		target_info = target_info,
	})
	assert(pipeline != nil)
	return pipeline
}

renderer_destroy_mesh :: proc(renderer: ^Renderer, mesh: ^GpuMesh) {
	if mesh.index_buffer != nil do sdl.ReleaseGPUBuffer(renderer.gpu, mesh.index_buffer)
	if mesh.vertex_buffer != nil do sdl.ReleaseGPUBuffer(renderer.gpu, mesh.vertex_buffer)
	mesh^ = {}
}

renderer_replace_mesh :: proc(renderer: ^Renderer, handle: MeshHandle, mesh: ^CpuMesh) {
	index := int(handle)
	assert(0 <= index && index < len(renderer.meshes))
	upload := renderer_begin_upload(renderer)
	new_mesh := renderer_upload_mesh_data(&upload, mesh)
	renderer_end_upload(&upload)
	renderer_destroy_mesh(renderer, &renderer.meshes[index])
	renderer.meshes[index] = new_mesh
}

renderer_begin_upload :: proc(renderer: ^Renderer) -> RendererUpload {
	cmd_buf := sdl.AcquireGPUCommandBuffer(renderer.gpu)
	assert(cmd_buf != nil)
	copy_pass := sdl.BeginGPUCopyPass(cmd_buf)
	assert(copy_pass != nil)
	return RendererUpload{
		renderer = renderer,
		cmd_buf = cmd_buf,
		copy_pass = copy_pass,
		transfer_buffers = make([dynamic]^sdl.GPUTransferBuffer, 0, 8, renderer.allocator),
	}
}

renderer_end_upload :: proc(upload: ^RendererUpload) {
	sdl.EndGPUCopyPass(upload.copy_pass)
	ok := sdl.SubmitGPUCommandBuffer(upload.cmd_buf)
	assert(ok)
	for transfer_buffer in upload.transfer_buffers {
		sdl.ReleaseGPUTransferBuffer(upload.renderer.gpu, transfer_buffer)
	}
	delete(upload.transfer_buffers)
	upload^ = {}
}

// Uploads one immutable mesh by staging vertex/index data through a shared startup upload pass.
renderer_upload_mesh :: proc(upload: ^RendererUpload, mesh: ^CpuMesh) -> MeshHandle {
	renderer := upload.renderer
	gpu_mesh := renderer_upload_mesh_data(upload, mesh)
	handle := MeshHandle(len(renderer.meshes))
	append(&renderer.meshes, gpu_mesh)
	renderer_update_static_stats(renderer)
	return handle
}

renderer_upload_mesh_data :: proc(upload: ^RendererUpload, mesh: ^CpuMesh) -> GpuMesh {
	renderer := upload.renderer
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
	append(&upload.transfer_buffers, transfer_buffer)

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

	sdl.UploadToGPUBuffer(
		upload.copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{buffer = gpu_mesh.vertex_buffer, offset = 0, size = u32(vertex_data_size)},
		false,
	)
	sdl.UploadToGPUBuffer(
		upload.copy_pass,
		{transfer_buffer = transfer_buffer, offset = u32(vertex_data_size)},
		{buffer = gpu_mesh.index_buffer, offset = 0, size = u32(index_data_size)},
		false,
	)
	return gpu_mesh
}

renderer_draw :: proc(
	renderer: ^Renderer,
	cmd_buf: ^sdl.GPUCommandBuffer,
	swapchain_tex: ^sdl.GPUTexture,
	globals: RenderPassGlobals,
	items: []RenderItem,
	debug_lines: []DebugLine,
	ui_commands: []UiCommand,
	viewport: [2]i32,
) {
	if swapchain_tex == nil do return
	renderer.stats.draw_count = 0
	renderer.stats.triangle_count = 0
	renderer_prepare_text(renderer, cmd_buf, ui_commands)
	renderer_prepare_debug_lines(renderer, cmd_buf, debug_lines)

	color_target := renderer_color_target(renderer, swapchain_tex, globals.environment)
	depth_target := sdl.GPUDepthStencilTargetInfo {
		texture          = renderer.depth_texture,
		clear_depth      = 1.0,
		load_op          = .CLEAR,
		store_op         = .DONT_CARE,
		stencil_load_op  = .DONT_CARE,
		stencil_store_op = .DONT_CARE,
	}

	render_pass := sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, &depth_target)
	renderer_draw_sky(renderer, cmd_buf, render_pass, globals)

	sdl.BindGPUGraphicsPipeline(render_pass, renderer.pipeline)
	fragment_uniforms := renderer_fragment_uniforms(globals.environment)
	sdl.PushGPUFragmentUniformData(cmd_buf, 0, &fragment_uniforms, size_of(fragment_uniforms))

	for item in items {
		mesh := renderer_mesh(renderer, item.mesh)
		vertex_buffer_binding := sdl.GPUBufferBinding{buffer = mesh.vertex_buffer, offset = 0}
		index_buffer_binding := sdl.GPUBufferBinding{buffer = mesh.index_buffer, offset = 0}
		sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)
		sdl.BindGPUIndexBuffer(render_pass, index_buffer_binding, ._32BIT)

		vertex_uniforms := WorldVertexUniforms {
			mvp = globals.proj * globals.view * item.model,
			model_view = globals.view * item.model,
		}
		sdl.PushGPUVertexUniformData(cmd_buf, 0, &vertex_uniforms, size_of(vertex_uniforms))
		sdl.DrawGPUIndexedPrimitives(render_pass, u32(mesh.index_count), 1, 0, 0, 0)
		renderer.stats.draw_count += 1
		renderer.stats.triangle_count += mesh.index_count / 3
	}
	renderer_draw_debug_lines(renderer, cmd_buf, render_pass, globals)
	renderer_draw_ui(renderer, cmd_buf, render_pass, ui_commands, viewport)
	renderer_draw_text(renderer, cmd_buf, render_pass, viewport)
	sdl.EndGPURenderPass(render_pass)
}

renderer_prepare_debug_lines :: proc(renderer: ^Renderer, cmd_buf: ^sdl.GPUCommandBuffer, lines: []DebugLine) {
	clear(&renderer.debug_line_vertices)
	if len(lines) == 0 do return
	line_count := min(len(lines), DEBUG_LINE_CAPACITY)
	for line in lines[:line_count] {
		append(&renderer.debug_line_vertices, VertexData{pos = line.from, color = line.color})
		append(&renderer.debug_line_vertices, VertexData{pos = line.to, color = line.color})
	}
	data_size := len(renderer.debug_line_vertices) * size_of(VertexData)
	transfer_ptr := transmute([^]byte)sdl.MapGPUTransferBuffer(renderer.gpu, renderer.debug_line_transfer_buffer, true)
	mem.copy(transfer_ptr[:], raw_data(renderer.debug_line_vertices[:]), data_size)
	sdl.UnmapGPUTransferBuffer(renderer.gpu, renderer.debug_line_transfer_buffer)
	copy_pass := sdl.BeginGPUCopyPass(cmd_buf)
	assert(copy_pass != nil)
	sdl.UploadToGPUBuffer(copy_pass, {transfer_buffer = renderer.debug_line_transfer_buffer, offset = 0}, {buffer = renderer.debug_line_vertex_buffer, offset = 0, size = u32(data_size)}, true)
	sdl.EndGPUCopyPass(copy_pass)
}

renderer_draw_debug_lines :: proc(renderer: ^Renderer, cmd_buf: ^sdl.GPUCommandBuffer, render_pass: ^sdl.GPURenderPass, globals: RenderPassGlobals) {
	if len(renderer.debug_line_vertices) == 0 do return
	sdl.BindGPUGraphicsPipeline(render_pass, renderer.line_pipeline)
	fragment_uniforms := renderer_fragment_uniforms(globals.environment)
	sdl.PushGPUFragmentUniformData(cmd_buf, 0, &fragment_uniforms, size_of(fragment_uniforms))
	vertex_uniforms := WorldVertexUniforms{mvp = globals.proj * globals.view, model_view = globals.view}
	sdl.PushGPUVertexUniformData(cmd_buf, 0, &vertex_uniforms, size_of(vertex_uniforms))
	vertex_buffer_binding := sdl.GPUBufferBinding{buffer = renderer.debug_line_vertex_buffer, offset = 0}
	sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)
	sdl.DrawGPUPrimitives(render_pass, u32(len(renderer.debug_line_vertices)), 1, 0, 0)
	renderer.stats.draw_count += 1
}

renderer_fragment_uniforms :: proc(environment: RenderEnvironment) -> FragmentUniforms {
	return {
		fog_color = environment.fog_color,
		sky_top_color = environment.sky_top_color,
		sky_horizon_color = environment.sky_horizon_color,
		fog_distances = environment.fog_distances,
	}
}

renderer_mesh :: proc(renderer: ^Renderer, handle: MeshHandle) -> ^GpuMesh {
	index := int(handle)
	assert(0 <= index && index < len(renderer.meshes))
	return &renderer.meshes[index]
}

renderer_update_static_stats :: proc(renderer: ^Renderer) {
	renderer.stats.mesh_count = len(renderer.meshes)
	renderer.stats.pipeline_count = int(renderer.pipeline != nil) + int(renderer.line_pipeline != nil) + int(renderer.sky_pipeline != nil) + int(renderer.ui_pipeline != nil) + int(renderer.text != nil && renderer.text.pipeline != nil)
}

renderer_log_stats :: proc(renderer: ^Renderer) {
	log.debugf(
		"Renderer stats: draws=%d triangles=%d meshes=%d pipelines=%d",
		renderer.stats.draw_count,
		renderer.stats.triangle_count,
		renderer.stats.mesh_count,
		renderer.stats.pipeline_count,
	)
}

renderer_color_target :: proc(
	renderer: ^Renderer,
	swapchain_tex: ^sdl.GPUTexture,
	environment: RenderEnvironment,
) -> sdl.GPUColorTargetInfo {
	color_target := sdl.GPUColorTargetInfo {
		load_op     = .CLEAR,
		clear_color = sdl.FColor(environment.fog_color),
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

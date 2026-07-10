package engine

import "core:math"
import sdl "vendor:sdl3"

UiVertexUniforms :: struct {
	rect:          [4]f32,
	color:         [4]f32,
	corner_radii:  [4]f32,
	border_widths: [4]f32,
	viewport:      [2]f32,
	padding:       [2]f32,
}

renderer_create_ui_pipeline :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	sample_count: sdl.GPUSampleCount,
) -> ^sdl.GPUGraphicsPipeline {
	vert_shader := load_shader(
		gpu,
		ui_vert_shader_code,
		.VERTEX,
		num_uniform_buffers = 1,
		entrypoint_name = "UiVertexMain",
	)
	frag_shader := load_shader(
		gpu,
		ui_frag_shader_code,
		.FRAGMENT,
		num_uniform_buffers = 1,
		entrypoint_name = "UiFragmentMain",
	)
	assert(vert_shader != nil)
	assert(frag_shader != nil)
	defer sdl.ReleaseGPUShader(gpu, vert_shader)
	defer sdl.ReleaseGPUShader(gpu, frag_shader)

	color_target_description := sdl.GPUColorTargetDescription {
		format = sdl.GetGPUSwapchainTextureFormat(gpu, window),
		blend_state = {
			src_color_blendfactor = .SRC_ALPHA,
			dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA,
			color_blend_op = .ADD,
			src_alpha_blendfactor = .ONE,
			dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA,
			alpha_blend_op = .ADD,
			enable_blend = true,
		},
	}
	target_info := sdl.GPUGraphicsPipelineTargetInfo {
		num_color_targets         = 1,
		color_target_descriptions = &color_target_description,
		depth_stencil_format      = .D32_FLOAT,
		has_depth_stencil_target  = true,
	}

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		{
			vertex_shader = vert_shader,
			fragment_shader = frag_shader,
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {
				cull_mode = .NONE,
				enable_depth_clip = true,
			},
			multisample_state = {
				sample_count = sample_count,
			},
			depth_stencil_state = {
				enable_depth_test = false,
				enable_depth_write = false,
			},
			target_info = target_info,
		},
	)
	assert(pipeline != nil)
	return pipeline
}

renderer_draw_ui :: proc(
	renderer: ^Renderer,
	cmd_buf: ^sdl.GPUCommandBuffer,
	render_pass: ^sdl.GPURenderPass,
	commands: []UiCommand,
	viewport: [2]i32,
) {
	if len(commands) == 0 do return

	full_scissor := sdl.Rect{x = 0, y = 0, w = viewport.x, h = viewport.y}
	sdl.SetGPUScissor(render_pass, full_scissor)
	sdl.BindGPUGraphicsPipeline(render_pass, renderer.ui_pipeline)

	for command in commands {
		switch command.kind {
		case .Rectangle:
			renderer_draw_ui_quad(renderer, cmd_buf, render_pass, command, viewport)
		case .Border:
			renderer_draw_ui_quad(renderer, cmd_buf, render_pass, command, viewport)
		case .Text:
			// Text commands are translated at the UI boundary; renderer text draw is added separately.
		case .ScissorStart:
			sdl.SetGPUScissor(render_pass, renderer_ui_scissor(command.bounds, viewport))
		case .ScissorEnd:
			sdl.SetGPUScissor(render_pass, full_scissor)
		}
	}

	sdl.SetGPUScissor(render_pass, full_scissor)
}

renderer_draw_ui_quad :: proc(
	renderer: ^Renderer,
	cmd_buf: ^sdl.GPUCommandBuffer,
	render_pass: ^sdl.GPURenderPass,
	command: UiCommand,
	viewport: [2]i32,
) {
	bounds := command.bounds
	if bounds.z <= 0 || bounds.w <= 0 do return

	uniforms := UiVertexUniforms {
		rect = bounds,
		color = command.color,
		corner_radii = command.corner_radii,
		border_widths = command.border_widths,
		viewport = {f32(viewport.x), f32(viewport.y)},
	}
	sdl.PushGPUVertexUniformData(cmd_buf, 0, &uniforms, size_of(uniforms))
	sdl.PushGPUFragmentUniformData(cmd_buf, 0, &uniforms, size_of(uniforms))
	sdl.DrawGPUPrimitives(render_pass, 6, 1, 0, 0)
	renderer.stats.draw_count += 1
	renderer.stats.triangle_count += 2
}

renderer_ui_scissor :: proc(bounds: [4]f32, viewport: [2]i32) -> sdl.Rect {
	x0 := max(i32(math.floor(bounds.x)), 0)
	y0 := max(i32(math.floor(bounds.y)), 0)
	x1 := min(i32(math.ceil(bounds.x + bounds.z)), viewport.x)
	y1 := min(i32(math.ceil(bounds.y + bounds.w)), viewport.y)
	return {x = x0, y = y0, w = max(x1 - x0, 0), h = max(y1 - y0, 0)}
}

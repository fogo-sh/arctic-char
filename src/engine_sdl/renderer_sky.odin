package engine_sdl

import "core:math/linalg"
import engine "../engine"
import sdl "vendor:sdl3"

SkyVertexUniforms :: struct {
	inv_view: matrix[4, 4]f32,
	inv_proj: matrix[4, 4]f32,
}

renderer_create_sky_pipeline :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	sample_count: sdl.GPUSampleCount,
) -> ^sdl.GPUGraphicsPipeline {
	vert_shader := load_shader(
		gpu,
		sky_vert_shader_code,
		.VERTEX,
		num_uniform_buffers = 2,
		entrypoint_name = "SkyVertexMain",
	)
	frag_shader := load_shader(
		gpu,
		sky_frag_shader_code,
		.FRAGMENT,
		num_uniform_buffers = 1,
		entrypoint_name = "SkyFragmentMain",
	)
	assert(vert_shader != nil)
	assert(frag_shader != nil)
	defer sdl.ReleaseGPUShader(gpu, vert_shader)
	defer sdl.ReleaseGPUShader(gpu, frag_shader)

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

renderer_draw_sky :: proc(
	renderer: ^Renderer,
	cmd_buf: ^sdl.GPUCommandBuffer,
	render_pass: ^sdl.GPURenderPass,
	globals: RenderPassGlobals,
) {
	sdl.BindGPUGraphicsPipeline(render_pass, renderer.sky_pipeline)
	vertex_uniforms := SkyVertexUniforms {
		inv_view = linalg.inverse(globals.view),
		inv_proj = linalg.inverse(globals.proj),
	}
	sdl.PushGPUVertexUniformData(cmd_buf, 1, &vertex_uniforms, size_of(vertex_uniforms))
	fragment_uniforms := renderer_fragment_uniforms(globals.environment)
	sdl.PushGPUFragmentUniformData(cmd_buf, 0, &fragment_uniforms, size_of(fragment_uniforms))
	sdl.DrawGPUPrimitives(render_pass, 3, 1, 0, 0)
}

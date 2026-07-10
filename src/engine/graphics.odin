package engine

import "base:runtime"
import "core:strings"
import sdl "vendor:sdl3"

when ODIN_OS == .Darwin {
	shader_format := sdl.GPUShaderFormat{.MSL}

	frag_shader_code := #load("../../shaders/msl/shader.msl.frag")
	vert_shader_code := #load("../../shaders/msl/shader.msl.vert")
	sky_frag_shader_code := #load("../../shaders/msl/sky.msl.frag")
	sky_vert_shader_code := #load("../../shaders/msl/sky.msl.vert")
} else when ODIN_OS == .Windows {
	shader_format := sdl.GPUShaderFormat{.DXIL}

	frag_shader_code := #load("../../shaders/dxil/shader.dxil.frag")
	vert_shader_code := #load("../../shaders/dxil/shader.dxil.vert")
	sky_frag_shader_code := #load("../../shaders/dxil/sky.dxil.frag")
	sky_vert_shader_code := #load("../../shaders/dxil/sky.dxil.vert")
} else {
	shader_format := sdl.GPUShaderFormat{.SPIRV}

	frag_shader_code := #load("../../shaders/spv/shader.spv.frag")
	vert_shader_code := #load("../../shaders/spv/shader.spv.vert")
	sky_frag_shader_code := #load("../../shaders/spv/sky.spv.frag")
	sky_vert_shader_code := #load("../../shaders/spv/sky.spv.vert")
}

load_shader :: proc(
	device: ^sdl.GPUDevice,
	code: []u8,
	stage: sdl.GPUShaderStage,
	num_uniform_buffers: u32,
	entrypoint_name: string,
) -> ^sdl.GPUShader {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	entrypoint := strings.clone_to_cstring(entrypoint_name, context.temp_allocator)
	return sdl.CreateGPUShader(
		device,
		{
			code_size = len(code),
			code = raw_data(code),
			entrypoint = entrypoint,
			format = shader_format,
			stage = stage,
			num_uniform_buffers = num_uniform_buffers,
			num_samplers = 0,
		},
	)
}

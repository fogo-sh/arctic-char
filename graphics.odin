package main

import "core:strings"
import "vendor:cgltf"
import sdl "vendor:sdl3"

when ODIN_OS == .Darwin {
	shader_entrypoint := "main0"
	shader_format := sdl.GPUShaderFormat{.MSL}

	frag_shader_code := #load("shaders/msl/shader.msl.frag")
	vert_shader_code := #load("shaders/msl/shader.msl.vert")

	ui_frag_shader_code := #load("shaders/msl/ui.msl.frag")
	ui_vert_shader_code := #load("shaders/msl/ui.msl.vert")
} else when ODIN_OS == .Windows {
	shader_entrypoint := "main0"
	shader_format := sdl.GPUShaderFormat{.DXIL}

	frag_shader_code := #load("shaders/dxil/shader.dxil.frag")
	vert_shader_code := #load("shaders/dxil/shader.dxil.vert")

	ui_frag_shader_code := #load("shaders/dxil/ui.dxil.frag")
	ui_vert_shader_code := #load("shaders/dxil/ui.dxil.vert")
} else {
	shader_entrypoint := "main"

	shader_format := sdl.GPUShaderFormat{.SPIRV}

	frag_shader_code := #load("shaders/spv/shader.spv.frag")
	vert_shader_code := #load("shaders/spv/shader.spv.vert")

	ui_frag_shader_code := #load("shaders/spv/ui.spv.frag")
	ui_vert_shader_code := #load("shaders/spv/ui.spv.vert")
}

create_msaa_textures :: proc(
	gpu: ^sdl.GPUDevice,
	window: ^sdl.Window,
	width: i32,
	height: i32,
) -> (
	msaa_color_texture: ^sdl.GPUTexture,
	msaa_depth_texture: ^sdl.GPUTexture,
) {
	msaa_color_texture = sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = sdl.GPUTextureType.D2,
			format = sdl.GetGPUSwapchainTextureFormat(gpu, window),
			usage = sdl.GPUTextureUsageFlags{.COLOR_TARGET},
			width = cast(u32)width,
			height = cast(u32)height,
			layer_count_or_depth = 1,
			num_levels = 1,
			sample_count = sdl.GPUSampleCount._4,
		},
	)
	sdl.SetGPUTextureName(gpu, msaa_color_texture, "MSAA Color Texture")
	assert(msaa_color_texture != nil)

	msaa_depth_texture = sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = sdl.GPUTextureType.D2,
			format = sdl.GPUTextureFormat.D32_FLOAT,
			usage = sdl.GPUTextureUsageFlags{.DEPTH_STENCIL_TARGET},
			width = cast(u32)width,
			height = cast(u32)height,
			layer_count_or_depth = 1,
			num_levels = 1,
			sample_count = sdl.GPUSampleCount._4,
		},
	)
	sdl.SetGPUTextureName(gpu, msaa_depth_texture, "MSAA Depth Texture")
	assert(msaa_depth_texture != nil)

	return msaa_color_texture, msaa_depth_texture
}

load_shader :: proc(
	device: ^sdl.GPUDevice,
	code: []u8,
	stage: sdl.GPUShaderStage,
	num_uniform_buffers: u32,
	num_samplers: u32,
) -> ^sdl.GPUShader {
	entrypoint := strings.clone_to_cstring(shader_entrypoint)
	defer delete(entrypoint)
	return sdl.CreateGPUShader(
		device,
		{
			code_size = len(code),
			code = raw_data(code),
			entrypoint = entrypoint,
			format = shader_format,
			stage = stage,
			num_uniform_buffers = num_uniform_buffers,
			num_samplers = num_samplers,
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
	defer delete(positions)
	_ = cgltf.accessor_unpack_floats(
		pos_accessor,
		raw_data(positions),
		uint(vertex_count * num_pos_components),
	)

	colors: []f32
	defer delete(colors)
	if col_accessor != nil {
		colors = make([]f32, vertex_count * num_col_components)
		_ = cgltf.accessor_unpack_floats(
			col_accessor,
			raw_data(colors),
			uint(vertex_count * num_col_components),
		)
	}

	uvs := make([]f32, vertex_count * num_uv_components)
	defer delete(uvs)
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

matrix4_orthographic_f32 :: proc(left, right, bottom, top, near, far: f32) -> matrix[4, 4]f32 {
	invRL := 1.0 / (right - left)
	invTB := 1.0 / (top - bottom)
	invFN := 1.0 / (far - near)
	return matrix[4, 4]f32{
		2 * invRL, 0, 0, -(right + left) * invRL, 
		0, 2 * invTB, 0, -(top + bottom) * invTB, 
		0, 0, -2 * invFN, -(far + near) * invFN, 
		0, 0, 0, 1, 
	}
}

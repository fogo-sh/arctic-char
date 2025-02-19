package main

import "core:mem"
import "vendor:cgltf"
import sdl "vendor:sdl3"

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

	vertex_count := pos_accessor^.count
	num_pos_components := cgltf.num_components(pos_accessor^.type)
	num_col_components := cgltf.num_components(col_accessor^.type)
	vertex_stride := (num_pos_components + num_col_components) * size_of(f32)

	idx_accessor := primitive.indices
	index_count := idx_accessor^.count

	expanded_size := u32(index_count * vertex_stride)
	expanded_data, err := mem.alloc(int(expanded_size))
	assert(err == nil)

	pos_data_size := vertex_count * num_pos_components * size_of(f32)
	pos_buffer, err2 := mem.alloc(int(pos_data_size))
	assert(err2 == nil)
	pos_slice := mem.slice_ptr(cast(^f32)pos_buffer, int(vertex_count * num_pos_components))
	_ = cgltf.accessor_unpack_floats(
		pos_accessor,
		raw_data(pos_slice),
		uint(vertex_count) * num_pos_components,
	)

	col_data_size := vertex_count * num_col_components * size_of(f32)
	col_buffer, err3 := mem.alloc(int(col_data_size))
	col_slice := mem.slice_ptr(cast(^f32)col_buffer, int(vertex_count * num_col_components))
	assert(err3 == nil)
	_ = cgltf.accessor_unpack_floats(
		col_accessor,
		raw_data(col_slice),
		uint(vertex_count) * num_col_components,
	)

	out_slice := mem.slice_ptr(
		cast(^f32)expanded_data,
		int(index_count * (num_pos_components + num_col_components)),
	)

	for i := uint(0); i < index_count; i += 3 {
		idx0 := cgltf.accessor_read_index(idx_accessor, i)
		idx1 := cgltf.accessor_read_index(idx_accessor, i + 1)
		idx2 := cgltf.accessor_read_index(idx_accessor, i + 2)

		base := i * (num_pos_components + num_col_components)
		for j := uint(0); j < num_pos_components; j += 1 {
			out_slice[base + j] = pos_slice[idx0 * num_pos_components + j]
		}
		for j := uint(0); j < num_col_components; j += 1 {
			out_slice[base + num_pos_components + j] = col_slice[idx0 * num_col_components + j]
		}

		base = (i + 1) * (num_pos_components + num_col_components)
		for j := uint(0); j < num_pos_components; j += 1 {
			out_slice[base + j] = pos_slice[idx1 * num_pos_components + j]
		}
		for j := uint(0); j < num_col_components; j += 1 {
			out_slice[base + num_pos_components + j] = col_slice[idx1 * num_col_components + j]
		}

		base = (i + 2) * (num_pos_components + num_col_components)
		for j := uint(0); j < num_pos_components; j += 1 {
			out_slice[base + j] = pos_slice[idx2 * num_pos_components + j]
		}
		for j := uint(0); j < num_col_components; j += 1 {
			out_slice[base + num_pos_components + j] = col_slice[idx2 * num_col_components + j]
		}
	}

	mem.free(pos_buffer)
	mem.free(col_buffer)

	transfer_buffer := sdl.CreateGPUTransferBuffer(
		gpu,
		{usage = .UPLOAD, size = expanded_size, props = 0},
	)
	tb_ptr := sdl.MapGPUTransferBuffer(gpu, transfer_buffer, false)
	mem.copy(tb_ptr, expanded_data, int(expanded_size))
	sdl.UnmapGPUTransferBuffer(gpu, transfer_buffer)

	vertex_buffer := sdl.CreateGPUBuffer(
		gpu,
		{usage = {.VERTEX, .INDEX}, size = expanded_size, props = 0},
	)

	copy_command_buffer := sdl.AcquireGPUCommandBuffer(gpu)
	copy_pass := sdl.BeginGPUCopyPass(copy_command_buffer)

	sdl.UploadToGPUBuffer(
		copy_pass,
		{transfer_buffer = transfer_buffer, offset = 0},
		{buffer = vertex_buffer, offset = 0, size = expanded_size},
		false,
	)
	sdl.EndGPUCopyPass(copy_pass)
	ok := sdl.SubmitGPUCommandBuffer(copy_command_buffer)
	assert(ok)

	mem.free(expanded_data)
	defer cgltf.free(data)

	return vertex_buffer, u32(index_count), u32(vertex_stride)
}

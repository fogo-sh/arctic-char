package main

import "core:mem"
import "vendor:cgltf"
import sdl "vendor:sdl3"

ModelDataInfo :: struct {
	data_ptr:      ^u8,
	size:          u32,
	vertex_count:  u32,
	vertex_stride: u32,
}

load_mesh_data :: proc(model_path: cstring) -> ModelDataInfo {
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
	assert(err3 == nil)
	col_slice := mem.slice_ptr(cast(^f32)col_buffer, int(vertex_count * num_col_components))
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
	defer cgltf.free(data)

	return ModelDataInfo {
		data_ptr = cast(^u8)expanded_data,
		size = expanded_size,
		vertex_count = u32(index_count),
		vertex_stride = u32(vertex_stride),
	}
}

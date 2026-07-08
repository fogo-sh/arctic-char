package main

import "core:strings"
import "vendor:cgltf"

Vec3 :: [3]f32
Color :: [4]f32

VertexData :: struct {
	pos:   Vec3,
	color: Color,
}

CpuMesh :: struct {
	vertices: []VertexData,
	indices:  []u16,
}

cpu_mesh_destroy :: proc(mesh: ^CpuMesh) {
	delete(mesh.vertices)
	delete(mesh.indices)
	mesh^ = {}
}

// Reads a GLB file into plain CPU arrays. GPU upload is intentionally separate
// so the file format step stays easy to inspect while learning the pipeline.
load_glb_mesh :: proc(model_path: string) -> CpuMesh {
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
	for &attr in primitive.attributes {
		#partial switch attr.type {
		case cgltf.attribute_type.position:
			pos_attr = &attr
		case cgltf.attribute_type.color:
			col_attr = &attr
		}
	}
	assert(pos_attr != nil)
	assert(primitive.indices != nil)

	pos_accessor := pos_attr.data
	col_accessor := col_attr != nil ? col_attr.data : nil
	idx_accessor := primitive.indices

	vertex_count := pos_accessor.count
	index_count := idx_accessor.count
	num_pos_components := cgltf.num_components(pos_accessor.type)
	num_col_components := col_accessor != nil ? cgltf.num_components(col_accessor.type) : 0

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

	vertices := make([]VertexData, vertex_count)
	for i := 0; i < int(vertex_count); i += 1 {
		pos_idx := i * int(num_pos_components)
		col_idx := i * int(num_col_components)

		color := Color{0.9, 0.78, 0.58, 1.0}
		if col_accessor != nil {
			color = Color{colors[col_idx + 0], colors[col_idx + 1], colors[col_idx + 2], 1.0}
		}

		vertices[i] = VertexData {
			pos   = Vec3{positions[pos_idx + 0], positions[pos_idx + 1], positions[pos_idx + 2]},
			color = color,
		}
	}

	indices := make([]u16, index_count)
	for i := 0; i < int(index_count); i += 1 {
		indices[i] = u16(cgltf.accessor_read_index(idx_accessor, uint(i)))
	}

	return CpuMesh{vertices = vertices, indices = indices}
}

create_ground_mesh :: proc() -> CpuMesh {
	vertices := make([]VertexData, 4)
	vertices[0] = {pos = {-50, 0, -50}, color = {0.18, 0.22, 0.20, 1}}
	vertices[1] = {pos = {50, 0, -50}, color = {0.18, 0.22, 0.20, 1}}
	vertices[2] = {pos = {50, 0, 50}, color = {0.16, 0.20, 0.18, 1}}
	vertices[3] = {pos = {-50, 0, 50}, color = {0.16, 0.20, 0.18, 1}}

	indices := make([]u16, 6)
	indices[0] = 0
	indices[1] = 2
	indices[2] = 1
	indices[3] = 0
	indices[4] = 3
	indices[5] = 2

	return CpuMesh{vertices = vertices, indices = indices}
}

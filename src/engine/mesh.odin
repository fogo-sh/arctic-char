package engine

import "base:runtime"
import "core:strings"
import "vendor:cgltf"

Vec3 :: [3]f32
Vec2 :: [2]f32
Color :: [4]f32

VertexData :: struct {
	pos:    Vec3,
	normal: Vec3,
	uv:     Vec2,
	color:  Color,
}

CpuMesh :: struct {
	vertices:  []VertexData,
	indices:   []u32,
	allocator: runtime.Allocator,
}

cpu_mesh_destroy :: proc(mesh: ^CpuMesh) {
	delete(mesh.vertices, mesh.allocator)
	delete(mesh.indices, mesh.allocator)
	mesh^ = {}
}

load_glb_mesh :: proc(fs: ^GameFS, qpath: string, allocator := context.allocator) -> CpuMesh {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	model_path, ok := game_fs_resolve(fs, qpath, context.temp_allocator)
	assert(ok)
	return load_glb_mesh_path(model_path, allocator)
}

// Reads a GLB file into plain CPU arrays. GPU upload is intentionally separate
// so the file format step stays easy to inspect while learning the pipeline.
load_glb_mesh_path :: proc(model_path: string, allocator := context.allocator) -> CpuMesh {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	model_path_cstr := strings.clone_to_cstring(model_path, context.temp_allocator)

	options: cgltf.options
	data, result := cgltf.parse_file(options, model_path_cstr)
	assert(result == .success)
	result = cgltf.load_buffers(options, data, model_path_cstr)
	assert(result == .success)
	defer cgltf.free(data)

	mesh := data.scene.nodes[0].mesh
	primitive := mesh.primitives[0]

	pos_attr: ^cgltf.attribute = nil
	normal_attr: ^cgltf.attribute = nil
	uv_attr: ^cgltf.attribute = nil
	col_attr: ^cgltf.attribute = nil
	for &attr in primitive.attributes {
		#partial switch attr.type {
		case cgltf.attribute_type.position:
			pos_attr = &attr
		case cgltf.attribute_type.normal:
			normal_attr = &attr
		case cgltf.attribute_type.texcoord:
			uv_attr = &attr
		case cgltf.attribute_type.color:
			col_attr = &attr
		}
	}
	assert(pos_attr != nil)
	assert(primitive.indices != nil)

	pos_accessor := pos_attr.data
	normal_accessor := normal_attr != nil ? normal_attr.data : nil
	uv_accessor := uv_attr != nil ? uv_attr.data : nil
	col_accessor := col_attr != nil ? col_attr.data : nil
	idx_accessor := primitive.indices

	vertex_count := pos_accessor.count
	index_count := idx_accessor.count
	num_pos_components := cgltf.num_components(pos_accessor.type)
	num_normal_components := normal_accessor != nil ? cgltf.num_components(normal_accessor.type) : 0
	num_uv_components := uv_accessor != nil ? cgltf.num_components(uv_accessor.type) : 0
	num_col_components := col_accessor != nil ? cgltf.num_components(col_accessor.type) : 0

	positions := make([]f32, vertex_count * num_pos_components, context.temp_allocator)
	_ = cgltf.accessor_unpack_floats(
		pos_accessor,
		raw_data(positions),
		uint(vertex_count * num_pos_components),
	)

	colors: []f32
	normals: []f32
	uvs: []f32
	if normal_accessor != nil {
		normals = make([]f32, vertex_count * num_normal_components, context.temp_allocator)
		_ = cgltf.accessor_unpack_floats(
			normal_accessor,
			raw_data(normals),
			uint(vertex_count * num_normal_components),
		)
	}
	if uv_accessor != nil {
		uvs = make([]f32, vertex_count * num_uv_components, context.temp_allocator)
		_ = cgltf.accessor_unpack_floats(
			uv_accessor,
			raw_data(uvs),
			uint(vertex_count * num_uv_components),
		)
	}
	if col_accessor != nil {
		colors = make([]f32, vertex_count * num_col_components, context.temp_allocator)
		_ = cgltf.accessor_unpack_floats(
			col_accessor,
			raw_data(colors),
			uint(vertex_count * num_col_components),
		)
	}

	vertices := make([]VertexData, vertex_count, allocator)
	for i := 0; i < int(vertex_count); i += 1 {
		pos_idx := i * int(num_pos_components)
		normal_idx := i * int(num_normal_components)
		uv_idx := i * int(num_uv_components)
		col_idx := i * int(num_col_components)

		color := Color{0.9, 0.78, 0.58, 1.0}
		if col_accessor != nil {
			color = Color{colors[col_idx + 0], colors[col_idx + 1], colors[col_idx + 2], num_col_components >= 4 ? colors[col_idx + 3] : 1.0}
		}
		normal := Vec3{0, 1, 0}
		if normal_accessor != nil {
			normal = Vec3{normals[normal_idx + 0], normals[normal_idx + 1], normals[normal_idx + 2]}
		}
		uv := Vec2{0, 0}
		if uv_accessor != nil {
			uv = Vec2{uvs[uv_idx + 0], uvs[uv_idx + 1]}
		}

		vertices[i] = VertexData {
			pos = Vec3{positions[pos_idx + 0], positions[pos_idx + 1], positions[pos_idx + 2]},
			normal = normal,
			uv = uv,
			color = color,
		}
	}

	indices := make([]u32, index_count, allocator)
	for i := 0; i < int(index_count); i += 1 {
		indices[i] = u32(cgltf.accessor_read_index(idx_accessor, uint(i)))
	}

	return CpuMesh{vertices = vertices, indices = indices, allocator = allocator}
}

package main

import "core:fmt"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:os"
import "core:reflect"
import "core:strconv"
import "core:strings"

BSP_VERSION :: 29

BSPHeader :: struct {
	version: i32,
}

BSPLump :: enum {
	ENTITIES,
	PLANES,
	TEXTURES,
	VERTICES,
	VISIBILITY,
	NODES,
	TEXINFO,
	FACES,
	LIGHTMAPS,
	CLIPNODES,
	LEAVES,
	LFACE,
	EDGES,
	LEDGES,
	MODELS,
	MAX_LUMPS,
}

BSPLumpHeader :: struct {
	offset: i32,
	length: i32,
}

BSPVertex :: struct {
	position: Vec3,
}

BSPPlane :: struct {
	normal: Vec3,
	dist:   f32,
	type:   i32,
}

BSPFace :: struct {
	plane_id:   u16,
	side:       u16,
	ledge_id:   i32,
	ledge_num:  u16,
	texinfo_id: u16,
	typelight:  u8,
	baselight:  u8,
	light:      [2]u8,
	lightmap:   i32,
}

BSPTexInfo :: struct {
	vectorS:    Vec3,
	distS:      f32,
	vectorT:    Vec3,
	distT:      f32,
	texture_id: u32,
	animated:   u32,
}

MipTexHeader :: struct {
	numtex: i32,
}

MipTex :: struct {
	name:    [16]byte,
	width:   u32,
	height:  u32,
	offset1: u32,
	offset2: u32,
	offset4: u32,
	offset8: u32,
}

BSPEdge :: struct {
	vertex0: u16,
	vertex1: u16,
}

BSPData :: struct {
	header:          BSPHeader,
	lumps:           [BSPLump.MAX_LUMPS]BSPLumpHeader,
	vertices:        []BSPVertex,
	faces:           []BSPFace,
	texinfos:        []BSPTexInfo,
	edges:           []BSPEdge,
	ledges:          []i32,
	render_vertices: []VertexData,
	render_indices:  []u16,
	textures:        []string,
}

load_bsp :: proc(bsp_path: string) -> (bsp_data: BSPData, ok: bool) {
	data, read_ok := os.read_entire_file(bsp_path)
	if !read_ok {
		log.error("Failed to read BSP file:", bsp_path)
		return {}, false
	}
	defer delete(data)

	if len(data) < size_of(BSPHeader) + size_of(BSPLumpHeader) * int(BSPLump.MAX_LUMPS) {
		log.error("BSP file too small")
		return {}, false
	}
	header := cast(^BSPHeader)&data[0]

	if header.version != BSP_VERSION {
		log.error("Unsupported BSP version:", header.version)
		return {}, false
	}

	bsp_data.header = header^

	lump_offset := size_of(BSPHeader)
	for i in 0 ..< int(BSPLump.MAX_LUMPS) {
		bsp_data.lumps[i] = (cast(^BSPLumpHeader)&data[lump_offset])^
		lump_offset += size_of(BSPLumpHeader)
	}

	{
		lump := bsp_data.lumps[BSPLump.VERTICES]
		vert_count := lump.length / size_of(BSPVertex)
		bsp_data.vertices = make([]BSPVertex, vert_count)

		for i in 0 ..< vert_count {
			offset := lump.offset + i32(i * size_of(BSPVertex))
			bsp_data.vertices[i] = (cast(^BSPVertex)&data[offset])^
		}
	}

	{
		lump := bsp_data.lumps[BSPLump.FACES]
		face_count := lump.length / size_of(BSPFace)
		bsp_data.faces = make([]BSPFace, face_count)

		for i in 0 ..< face_count {
			offset := lump.offset + i32(i * size_of(BSPFace))
			bsp_data.faces[i] = (cast(^BSPFace)&data[offset])^
		}
	}

	{
		lump := bsp_data.lumps[BSPLump.TEXINFO]
		texinfo_count := lump.length / size_of(BSPTexInfo)
		bsp_data.texinfos = make([]BSPTexInfo, texinfo_count)

		for i in 0 ..< texinfo_count {
			offset := lump.offset + i32(i * size_of(BSPTexInfo))
			bsp_data.texinfos[i] = (cast(^BSPTexInfo)&data[offset])^
		}
	}

	{
		lump := bsp_data.lumps[BSPLump.EDGES]
		edge_count := lump.length / size_of(BSPEdge)
		bsp_data.edges = make([]BSPEdge, edge_count)

		for i in 0 ..< edge_count {
			offset := lump.offset + i32(i * size_of(BSPEdge))
			bsp_data.edges[i] = (cast(^BSPEdge)&data[offset])^
		}
	}

	{
		lump := bsp_data.lumps[BSPLump.LEDGES]
		ledge_count := lump.length / size_of(i32)
		bsp_data.ledges = make([]i32, ledge_count)

		for i in 0 ..< ledge_count {
			offset := lump.offset + i32(i * size_of(i32))
			bsp_data.ledges[i] = (cast(^i32)&data[offset])^
		}
	}

	{
		lump := bsp_data.lumps[BSPLump.TEXTURES]
		if lump.length > 0 {
			mip_header := cast(^MipTexHeader)&data[lump.offset]
			num_textures := mip_header.numtex

			bsp_data.textures = make([]string, num_textures)

			texture_offsets := make([]i32, num_textures)
			defer delete(texture_offsets)

			for i in 0 ..< num_textures {
				offset_ptr := cast(^i32)&data[lump.offset + size_of(MipTexHeader) + i * size_of(i32)]
				texture_offsets[i] = offset_ptr^
			}

			for i in 0 ..< num_textures {
				if texture_offsets[i] == -1 {
					bsp_data.textures[i] = ""
					continue
				}

				tex_offset := lump.offset + texture_offsets[i]
				mip_tex := cast(^MipTex)&data[tex_offset]

				name_len := 0
				for j in 0 ..< 16 {
					if mip_tex.name[j] == 0 {
						break
					}
					name_len += 1
				}

				name_bytes := mip_tex.name[:name_len]
				bsp_data.textures[i] = strings.clone_from_bytes(name_bytes)
			}
		}
	}

	convert_bsp_to_model(&bsp_data)

	return bsp_data, true
}

convert_bsp_to_model :: proc(bsp_data: ^BSPData) {
	total_vertices := 0
	total_indices := 0

	for face in bsp_data.faces {
		n_indices := (face.ledge_num - 2) * 3
		total_indices += int(n_indices)
		total_vertices += int(face.ledge_num)
	}

	bsp_data.render_vertices = make([]VertexData, total_vertices)
	bsp_data.render_indices = make([]u16, total_indices)

	vertex_offset := 0
	index_offset := 0

	for face in bsp_data.faces {
		for i in 0 ..< int(face.ledge_num) {
			edge_idx := bsp_data.ledges[face.ledge_id + i32(i)]

			vertex_idx: i32
			if edge_idx >= 0 {
				vertex_idx = i32(bsp_data.edges[edge_idx].vertex0)
			} else {
				vertex_idx = i32(bsp_data.edges[-edge_idx].vertex1)
			}

			bsp_vertex := bsp_data.vertices[vertex_idx]

			transformed_pos := Vec3 {
				bsp_vertex.position.x,
				bsp_vertex.position.z,
				-bsp_vertex.position.y,
			}

			texinfo := bsp_data.texinfos[face.texinfo_id]

			s := linalg.dot(bsp_vertex.position, texinfo.vectorS) + texinfo.distS
			t := linalg.dot(bsp_vertex.position, texinfo.vectorT) + texinfo.distT

			texture_scale := 1.0 / 32.0
			s *= f32(texture_scale)
			t *= f32(texture_scale)

			t = 1.0 - t

			texture_name := bsp_data.textures[texinfo.texture_id]
			first_char := strings.to_upper(string(texture_name[0:1]))
			defer delete(first_char)
			rest_of_name := texture_name[1:]
			modified_texture_name := fmt.tprintf("%s%s", first_char, rest_of_name)
			texture_name_enum, ok := reflect.enum_from_name(Texture_Name, modified_texture_name)
			assert(ok)

			texture_index := f32(texture_name_enum)

			assert(texture_index != -1)

			fmt.printf(
				"Texture: %s -> %s (enum: %v, index: %v)\n",
				texture_name,
				modified_texture_name,
				texture_name_enum,
				texture_index,
			)

			uvw: [3]f32 = {s, t, texture_index}

			bsp_data.render_vertices[vertex_offset + i] = VertexData {
				pos   = transformed_pos,
				uvw   = uvw,
				color = {1.0, 1.0, 1.0, 1.0},
			}
		}

		for i in 0 ..< int(face.ledge_num - 2) {
			bsp_data.render_indices[index_offset + i * 3 + 0] = u16(vertex_offset)
			bsp_data.render_indices[index_offset + i * 3 + 1] = u16(vertex_offset + i + 2)
			bsp_data.render_indices[index_offset + i * 3 + 2] = u16(vertex_offset + i + 1)
		}

		index_offset += int((face.ledge_num - 2) * 3)
		vertex_offset += int(face.ledge_num)
	}
}

free_bsp_data :: proc(bsp_data: ^BSPData) {
	delete(bsp_data.vertices)
	delete(bsp_data.faces)
	delete(bsp_data.texinfos)
	delete(bsp_data.edges)
	delete(bsp_data.ledges)

	for texture_name in bsp_data.textures {
		delete(texture_name)
	}
	delete(bsp_data.textures)
}

bsp_to_model :: proc(bsp_data: ^BSPData) -> (vertices: []VertexData, indices: []u16) {
	return bsp_data.render_vertices[:], bsp_data.render_indices[:]
}

test_bsp_data :: proc() {
	bsp_data, ok := load_bsp("./assets/maps/test.bsp")
	assert(ok)

	fmt.println("BSP file loaded successfully")
	fmt.printf(
		"Vertices: %d, Faces: %d, Textures: %d\n",
		len(bsp_data.vertices),
		len(bsp_data.faces),
		len(bsp_data.texinfos),
	)
	fmt.printf(
		"Render vertices: %d, Render indices: %d\n",
		len(bsp_data.render_vertices),
		len(bsp_data.render_indices),
	)

	fmt.println("Texture names:")
	for name, i in bsp_data.textures {
		fmt.printf("  %d: %s\n", i, name)
	}

	defer free_bsp_data(&bsp_data)
}

package main

import "core:fmt"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:os"
import "core:strconv"
import "core:strings"

MapFace :: struct {
	plane_points: [3]Vec3,
	texture:      string,
	u_offset:     f32,
	v_offset:     f32,
	rotation:     f32,
	u_scale:      f32,
	v_scale:      f32,
}

MapBrush :: struct {
	faces: [dynamic]MapFace,
}

MapEntity :: struct {
	properties: map[string]string,
	brushes:    [dynamic]MapBrush,
}

MapData :: struct {
	entities: [dynamic]MapEntity,
}

load_map_data :: proc(map_path: string) -> (map_data: MapData, map_ok: bool) {
	data, ok := os.read_entire_file(map_path)
	if !ok {
		return
	}
	defer delete(data)

	parse_state :: enum {
		NONE,
		ENTITY,
		BRUSH,
	}

	state := parse_state.NONE

	current_entity: MapEntity
	current_brush: MapBrush

	it := string(data)

	for raw_line in strings.split_lines_iterator(&it) {
		line := strings.trim_space(raw_line)

		if len(line) == 0 || strings.has_prefix(line, "//") {
			continue
		}

		if line == "{" {
			if state == parse_state.NONE {
				state = parse_state.ENTITY
				current_entity.properties = make(map[string]string)
				current_entity.brushes = make([dynamic]MapBrush)
			} else if state == parse_state.ENTITY {
				state = parse_state.BRUSH
				current_brush.faces = make([dynamic]MapFace)
			}
		} else if line == "}" {
			if state == parse_state.BRUSH {
				append(&current_entity.brushes, current_brush)
				current_brush = {}
				state = parse_state.ENTITY
			} else if state == parse_state.ENTITY {
				append(&map_data.entities, current_entity)
				current_entity = {}
				state = parse_state.NONE
			}
		} else if state == parse_state.ENTITY && strings.contains(line, "\"") {
			parts := strings.split(line, "\"")
			defer delete(parts)
			if len(parts) >= 3 {
				key := parts[1]
				value := parts[3]
				current_entity.properties[key] = value
			}
		} else if state == parse_state.BRUSH && strings.has_prefix(line, "(") {
			face: MapFace

			plane_points_str := strings.split(line, ")")
			defer delete(plane_points_str)
			if len(plane_points_str) >= 4 {
				for i in 0 ..< 3 {
					point_str := strings.trim_prefix(strings.trim_space(plane_points_str[i]), "(")
					coords := strings.split(point_str, " ")

					filtered_coords := make([dynamic]string)
					defer delete(filtered_coords)

					for coord in coords {
						if coord != "" {
							append(&filtered_coords, coord)
						}
					}

					delete(coords)

					coords = filtered_coords[:]
					if len(coords) >= 3 {
						x, ok1 := strconv.parse_f32(coords[0])
						assert(ok1, "Failed to parse x coordinate")
						y, ok2 := strconv.parse_f32(coords[1])
						assert(ok2, "Failed to parse y coordinate")
						z, ok3 := strconv.parse_f32(coords[2])
						assert(ok3, "Failed to parse z coordinate")
						face.plane_points[i] = Vec3{x, y, z}
					}
				}

				remaining := strings.trim_space(plane_points_str[3])
				params := strings.split(remaining, " ")
				defer delete(params)
				if len(params) >= 6 {
					face.texture = params[0]
					u_offset, ok4 := strconv.parse_f32(params[1])
					assert(ok4, "Failed to parse u_offset")
					face.u_offset = u_offset

					v_offset, ok5 := strconv.parse_f32(params[2])
					assert(ok5, "Failed to parse v_offset")
					face.v_offset = v_offset

					rotation, ok6 := strconv.parse_f32(params[3])
					assert(ok6, "Failed to parse rotation")
					face.rotation = rotation

					u_scale, ok7 := strconv.parse_f32(params[4])
					assert(ok7, "Failed to parse u_scale")
					face.u_scale = u_scale

					v_scale, ok8 := strconv.parse_f32(params[5])
					assert(ok8, "Failed to parse v_scale")
					face.v_scale = v_scale
				}

				append(&current_brush.faces, face)
			}
		}
	}

	return map_data, true
}

free_map_data :: proc(map_data: MapData) {
	for entity in map_data.entities {
		delete(entity.properties)

		for brush in entity.brushes {
			delete(brush.faces)
		}

		delete(entity.brushes)
	}

	delete(map_data.entities)
}

test_map_data :: proc() {
	map_data, ok := load_map_data("./assets/maps/test.map")
	assert(ok)
	fmt.println("Map Data Structure:")
	for entity, entity_idx in map_data.entities {
		fmt.printf("Entity %d:\n", entity_idx)
		fmt.printf("  Properties (%d):\n", len(entity.properties))

		for key, value in entity.properties {
			fmt.printf("    %s = %s\n", key, value)
		}

		fmt.printf("  Brushes (%d):\n", len(entity.brushes))

		for brush, brush_idx in entity.brushes {
			fmt.printf("    Brush %d:\n", brush_idx)
			fmt.printf("      Faces (%d):\n", len(brush.faces))

			for face, face_idx in brush.faces {
				fmt.printf("        Face %d:\n", face_idx)
				fmt.printf(
					"          Plane: (%f, %f, %f) (%f, %f, %f) (%f, %f, %f)\n",
					face.plane_points[0].x,
					face.plane_points[0].y,
					face.plane_points[0].z,
					face.plane_points[1].x,
					face.plane_points[1].y,
					face.plane_points[1].z,
					face.plane_points[2].x,
					face.plane_points[2].y,
					face.plane_points[2].z,
				)
				fmt.printf("          Texture: %s\n", face.texture)
				fmt.printf(
					"          UV Mapping: offset=(%f, %f), rotation=%f, scale=(%f, %f)\n",
					face.u_offset,
					face.v_offset,
					face.rotation,
					face.u_scale,
					face.v_scale,
				)
			}
		}
	}
	defer free_map_data(map_data)
}

map_to_model :: proc(map_data: MapData) -> ModelData {
	vertices := make([dynamic]VertexData)
	indices := make([dynamic]u16)

	for entity in map_data.entities {
		for brush in entity.brushes {
			for face in brush.faces {
				p1, p2, p3 := face.plane_points[0], face.plane_points[1], face.plane_points[2]

				edge1 := p2 - p1
				edge2 := p3 - p1
				normal := linalg.normalize(linalg.cross(edge1, edge2))

				axis := 0
				if abs(normal.y) > abs(normal.x) {
					axis = 1
				}
				if abs(normal.z) > abs(normal[axis]) {
					axis = 2
				}

				v_offset := cast(u16)len(vertices)

				size := f32(64.0)

				quad_points: [4]Vec3
				if axis == 0 {
					quad_points[0] = {p1.x, p1.y - size, p1.z - size}
					quad_points[1] = {p1.x, p1.y + size, p1.z - size}
					quad_points[2] = {p1.x, p1.y + size, p1.z + size}
					quad_points[3] = {p1.x, p1.y - size, p1.z + size}
				} else if axis == 1 {
					quad_points[0] = {p1.x - size, p1.y, p1.z - size}
					quad_points[1] = {p1.x + size, p1.y, p1.z - size}
					quad_points[2] = {p1.x + size, p1.y, p1.z + size}
					quad_points[3] = {p1.x - size, p1.y, p1.z + size}
				} else {
					quad_points[0] = {p1.x - size, p1.y - size, p1.z}
					quad_points[1] = {p1.x + size, p1.y - size, p1.z}
					quad_points[2] = {p1.x + size, p1.y + size, p1.z}
					quad_points[3] = {p1.x - size, p1.y + size, p1.z}
				}

				for i in 0 ..< 4 {
					u := face.u_offset + f32(i % 2) * face.u_scale
					v := face.v_offset + f32(i / 2) * face.v_scale

					vertex := VertexData {
						pos   = quad_points[i],
						color = {1.0, 1.0, 1.0, 1.0},
						uv    = {u, v},
					}
					append(&vertices, vertex)
				}

				append(&indices, v_offset)
				append(&indices, v_offset + 1)
				append(&indices, v_offset + 2)

				append(&indices, v_offset)
				append(&indices, v_offset + 2)
				append(&indices, v_offset + 3)
			}
		}
	}

	return ModelData{vertices = vertices[:], indices = indices[:]}
}

free_model_data :: proc(model: ModelData) {
	delete(model.vertices)
	delete(model.indices)
}

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

free_map_data :: proc(map_data: ^MapData) {
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

	debug_print_map_data(&map_data)

	defer free_map_data(&map_data)
}

debug_print_map_data :: proc(map_data: ^MapData) {
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
}

map_to_model :: proc(map_data: ^MapData) -> (vertices: []VertexData, indices: []u16) {
	fmt.println("Starting map_to_model conversion...")

	vertex_count := 0
	index_count := 0

	brush_count := 0
	face_count := 0
	valid_face_count := 0

	for entity in map_data.entities {
		for brush in entity.brushes {
			brush_count += 1

			planes := make([]Plane, len(brush.faces))
			defer delete(planes)

			for face, i in brush.faces {
				face_count += 1
				p1, p2, p3 := face.plane_points[0], face.plane_points[1], face.plane_points[2]
				edge1 := p2 - p1
				edge2 := p3 - p1
				normal := linalg.normalize(linalg.cross(edge1, edge2))
				d := -linalg.dot(normal, p1)
				planes[i] = Plane{normal, d}

				fmt.printf("Face %d plane: normal=%v, distance=%f\n", i, normal, d)
			}

			for i := 0; i < len(brush.faces); i += 1 {
				polygon := calculate_face_polygon(planes, i)

				if len(polygon) >= 3 {
					valid_face_count += 1
					vertex_count += len(polygon)
					index_count += (len(polygon) - 2) * 3
					fmt.printf(
						"Valid polygon found for face %d with %d vertices\n",
						i,
						len(polygon),
					)
				} else {
					fmt.printf("Invalid or clipped away polygon for face %d\n", i)
				}

				delete(polygon)
			}
		}
	}

	fmt.printf(
		"Brush count: %d, Face count: %d, Valid face count: %d\n",
		brush_count,
		face_count,
		valid_face_count,
	)
	fmt.printf("Total vertices: %d, Total indices: %d\n", vertex_count, index_count)

	assert(vertex_count > 0, "No valid geometry found")

	vertices = make([]VertexData, vertex_count)
	indices = make([]u16, index_count)

	vertex_idx := 0
	index_idx := 0

	for entity in map_data.entities {
		for brush in entity.brushes {
			// Calculate planes again
			planes := make([]Plane, len(brush.faces))
			defer delete(planes)

			for face, i in brush.faces {
				p1, p2, p3 := face.plane_points[0], face.plane_points[1], face.plane_points[2]
				edge1 := p2 - p1
				edge2 := p3 - p1
				normal := linalg.normalize(linalg.cross(edge1, edge2))
				d := -linalg.dot(normal, p1)
				planes[i] = Plane{normal, d}
			}

			// Process each face
			for i := 0; i < len(brush.faces); i += 1 {
				face := brush.faces[i]
				polygon := calculate_face_polygon(planes, i)

				if len(polygon) >= 3 {
					// Add vertices
					v_offset := u16(vertex_idx)
					normal := planes[i].normal

					// Calculate texture axes based on normal
					tex_s, tex_t := calculate_texture_axes(normal)

					// Add all vertices of the polygon
					for j := 0; j < len(polygon); j += 1 {
						point := polygon[j]

						// Calculate texture coordinates
						u := face.u_offset + linalg.dot(point, tex_s) / face.u_scale
						v := face.v_offset + linalg.dot(point, tex_t) / face.v_scale

						// Apply rotation if needed
						if face.rotation != 0 {
							rot_rad := face.rotation * math.PI / 180.0
							s := math.sin(rot_rad)
							c := math.cos(rot_rad)
							u_old, v_old := u, v
							u = u_old * c - v_old * s
							v = u_old * s + v_old * c
						}

						vertices[vertex_idx] = VertexData {
							pos   = point,
							color = {1.0, 1.0, 1.0, 1.0},
							uv    = {u, v},
						}
						vertex_idx += 1
					}

					// Triangulate using a fan
					for j := 0; j < len(polygon) - 2; j += 1 {
						indices[index_idx] = v_offset
						indices[index_idx + 1] = v_offset + u16(j + 1)
						indices[index_idx + 2] = v_offset + u16(j + 2)
						index_idx += 3
					}
				}

				delete(polygon)
			}
		}
	}

	return vertices, indices
}

calculate_face_polygon :: proc(planes: []Plane, face_idx: int) -> []Vec3 {
	if len(planes) == 0 {
		return make([]Vec3, 0)
	}

	face_plane := planes[face_idx]

	normal := face_plane.normal

	u, v: Vec3

	if math.abs(normal.x) <= math.abs(normal.y) && math.abs(normal.x) <= math.abs(normal.z) {
		u = {0, -normal.z, normal.y}
	} else if math.abs(normal.y) <= math.abs(normal.z) {
		u = {-normal.z, 0, normal.x}
	} else {
		u = {-normal.y, normal.x, 0}
	}

	u = linalg.normalize(u)
	v = linalg.normalize(linalg.cross(normal, u))

	point_on_plane := normal * -face_plane.distance

	size := f32(1000.0)
	polygon := make([]Vec3, 4)
	polygon[0] = point_on_plane + u * size + v * size
	polygon[1] = point_on_plane - u * size + v * size
	polygon[2] = point_on_plane - u * size - v * size
	polygon[3] = point_on_plane + u * size - v * size

	for i := 0; i < len(planes); i += 1 {
		if i == face_idx do continue

		plane := planes[i]

		clip_plane := Plane{-plane.normal, -plane.distance}

		new_polygon := clip_polygon_to_plane(polygon, clip_plane)
		delete(polygon)
		polygon = new_polygon

		if len(polygon) < 3 {
			delete(polygon)
			return make([]Vec3, 0)
		}
	}

	return polygon
}

Plane :: struct {
	normal:   Vec3,
	distance: f32,
}

clip_polygon_to_plane :: proc(polygon: []Vec3, plane: Plane) -> []Vec3 {
	if len(polygon) == 0 do return make([]Vec3, 0)

	result := make([dynamic]Vec3)

	for i := 0; i < len(polygon); i += 1 {
		current := polygon[i]
		next := polygon[(i + 1) % len(polygon)]

		current_inside := is_point_inside_plane(current, plane)
		next_inside := is_point_inside_plane(next, plane)

		if current_inside {
			append(&result, current)
		}

		if current_inside != next_inside {
			t := plane_line_intersection(current, next, plane)
			intersection := current + (next - current) * t
			append(&result, intersection)
		}
	}

	if len(result) < 3 {
		delete(result)
		return make([]Vec3, 0)
	}

	final_result := result[:]
	return final_result
}

is_point_inside_plane :: proc(point: Vec3, plane: Plane) -> bool {
	return linalg.dot(plane.normal, point) + plane.distance <= 0
}

plane_line_intersection :: proc(p1, p2: Vec3, plane: Plane) -> f32 {
	normal := plane.normal

	dot1 := linalg.dot(normal, p1) + plane.distance
	dot2 := linalg.dot(normal, p2) + plane.distance

	t := -dot1 / (dot2 - dot1)
	return t
}

calculate_texture_axes :: proc(normal: Vec3) -> (s_axis: Vec3, t_axis: Vec3) {
	ax, ay, az := math.abs(normal.x), math.abs(normal.y), math.abs(normal.z)

	if az >= ax && az >= ay {
		s_axis = {1, 0, 0}
		t_axis = {0, -1, 0}
	} else if ay >= ax {
		s_axis = {1, 0, 0}
		t_axis = {0, 0, -1}
	} else {
		s_axis = {0, 1, 0}
		t_axis = {0, 0, -1}
	}

	return s_axis, t_axis
}

free_model_data :: proc(model: ^ModelData) {
	delete(model.vertices)
	delete(model.indices)
}

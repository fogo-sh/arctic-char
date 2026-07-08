package main

import "base:runtime"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:strconv"

MapProperty :: struct {
	key:   string,
	value: string,
}

MapEntity :: struct {
	properties: [dynamic]MapProperty,
	brushes:    [dynamic]MapBrush,
}

MapBrush :: struct {
	faces: [dynamic]MapFace,
}

MapFace :: struct {
	points: [3]Vec3,
	normal: Vec3,
	d:      f32,
}

QuakeMap :: struct {
	allocator: runtime.Allocator,
	source:    []byte,
	entities:  [dynamic]MapEntity,
}

MapPlayerSpawn :: struct {
	position: Vec3,
	yaw:      f32,
}

quake_map_load :: proc(fs: ^GameFS, qpath: string, allocator := context.allocator) -> QuakeMap {
	data, ok := game_fs_read_file(fs, qpath, allocator)
	assert(ok)

	qmap := QuakeMap{
		allocator = allocator,
		source = data,
		entities = make([dynamic]MapEntity, 0, 16, allocator),
	}
	quake_map_parse_entities(&qmap)
	brush_count, face_count := quake_map_count_brush_data(&qmap)
	log.debugf("Loaded map: %s entities=%d brushes=%d faces=%d", qpath, len(qmap.entities), brush_count, face_count)
	return qmap
}

quake_map_destroy :: proc(qmap: ^QuakeMap) {
	for &entity in qmap.entities {
		delete(entity.properties)
		for &brush in entity.brushes {
			delete(brush.faces)
		}
		delete(entity.brushes)
	}
	delete(qmap.entities)
	delete(qmap.source)
	qmap^ = {}
}

quake_map_find_player_spawn :: proc(qmap: ^QuakeMap) -> (spawn: MapPlayerSpawn, ok: bool) {
	for &entity in qmap.entities {
		classname, has_classname := map_entity_property(&entity, "classname")
		if !has_classname || classname != "player" {
			continue
		}

		origin, has_origin := map_entity_property(&entity, "origin")
		if !has_origin {
			continue
		}

		position, origin_ok := quake_map_parse_origin(origin)
		if !origin_ok {
			continue
		}

		angle: f32
		if angle_text, has_angle := map_entity_property(&entity, "angle"); has_angle {
			angle, _ = strconv.parse_f32(angle_text)
		}

		spawn.position = position
		spawn.yaw = linalg.to_radians(angle + 180)
		return spawn, true
	}
	return {}, false
}

map_entity_property :: proc(entity: ^MapEntity, key: string) -> (value: string, ok: bool) {
	for property in entity.properties {
		if property.key == key {
			return property.value, true
		}
	}
	return "", false
}

quake_map_parse_origin :: proc(text: string) -> (position: Vec3, ok: bool) {
	x, n0, ok_x := quake_map_parse_f32_token(text, 0)
	y, n1, ok_y := quake_map_parse_f32_token(text, n0)
	z, _, ok_z := quake_map_parse_f32_token(text, n1)
	if !(ok_x && ok_y && ok_z) {
		return {}, false
	}

	// FuncGodot converts id-tech coordinates to Y-up OpenGL as {y, z, x}.
	return quake_map_to_world({x, y, z}), true
}

quake_map_to_world :: proc(position: Vec3) -> Vec3 {
	return Vec3{position.y, position.z, position.x} * QU_TO_M
}

quake_map_parse_f32_token :: proc(text: string, start: int) -> (value: f32, next: int, ok: bool) {
	i := quake_map_skip_inline_space(text, start)
	begin := i
	for i < len(text) && !quake_map_is_space(text[i]) {
		i += 1
	}
	if begin == i {
		return 0, i, false
	}
	value, ok = strconv.parse_f32(text[begin:i])
	return value, i, ok
}

quake_map_parse_entities :: proc(qmap: ^QuakeMap) {
	source := string(qmap.source)
	i := 0
	depth := 0
	entity: MapEntity
	brush: MapBrush

	for i < len(source) {
		i = quake_map_skip_ignored(source, i)
		if i >= len(source) {
			break
		}

		switch source[i] {
		case '{':
			depth += 1
			if depth == 1 {
				entity = MapEntity{
					properties = make([dynamic]MapProperty, 0, 8, qmap.allocator),
					brushes = make([dynamic]MapBrush, 0, 8, qmap.allocator),
				}
			} else if depth == 2 {
				brush = MapBrush{faces = make([dynamic]MapFace, 0, 6, qmap.allocator)}
			}
			i += 1

		case '}':
			if depth == 2 {
				append(&entity.brushes, brush)
				brush = {}
			} else if depth == 1 {
				append(&qmap.entities, entity)
				entity = {}
			}
			depth -= 1
			i += 1

		case '"':
			if depth == 1 {
				key, next, key_ok := quake_map_parse_quoted(source, i)
				if key_ok {
					next = quake_map_skip_inline_space(source, next)
					value, value_next, value_ok := quake_map_parse_quoted(source, next)
					if value_ok {
						append(&entity.properties, MapProperty{key = key, value = value})
						i = value_next
						continue
					}
				}
			}
			i = quake_map_skip_line(source, i)

		case '(':
			if depth == 2 {
				if face, ok := quake_map_parse_face(source, i); ok {
					append(&brush.faces, face)
				}
			}
			i = quake_map_skip_line(source, i)

		case:
			i = quake_map_skip_line(source, i)
		}
	}
}

quake_map_count_brush_data :: proc(qmap: ^QuakeMap) -> (brush_count, face_count: int) {
	for &entity in qmap.entities {
		brush_count += len(entity.brushes)
		for &brush in entity.brushes {
			face_count += len(brush.faces)
		}
	}
	return
}

quake_map_parse_face :: proc(source: string, start: int) -> (face: MapFace, ok: bool) {
	p0, next0, ok0 := quake_map_parse_point(source, start)
	p1, next1, ok1 := quake_map_parse_point(source, next0)
	p2, _, ok2 := quake_map_parse_point(source, next1)
	if !(ok0 && ok1 && ok2) {
		return {}, false
	}

	p0 = quake_map_to_world(p0)
	p1 = quake_map_to_world(p1)
	p2 = quake_map_to_world(p2)
	// MAP plane triples use id/QBSP winding. The normal points out of the brush;
	// the solid volume is the back side of every face plane.
	normal := linalg.normalize0(linalg.cross(p0 - p1, p2 - p1))
	return MapFace{points = {p0, p1, p2}, normal = normal, d = linalg.dot(normal, p0)}, true
}

quake_map_parse_point :: proc(source: string, start: int) -> (point: Vec3, next: int, ok: bool) {
	i := quake_map_skip_inline_space(source, start)
	if i >= len(source) || source[i] != '(' {
		return {}, i, false
	}
	i += 1

	x, n0, ok_x := quake_map_parse_f32_token(source, i)
	y, n1, ok_y := quake_map_parse_f32_token(source, n0)
	z, n2, ok_z := quake_map_parse_f32_token(source, n1)
	i = quake_map_skip_inline_space(source, n2)
	if i >= len(source) || source[i] != ')' || !(ok_x && ok_y && ok_z) {
		return {}, i, false
	}
	return Vec3{x, y, z}, i + 1, true
}

quake_map_face_color :: proc(face: ^MapFace) -> Color {
	n := face.normal
	return Color{
		0.24 + math.abs(n.x) * 0.30,
		0.24 + math.abs(n.y) * 0.34,
		0.24 + math.abs(n.z) * 0.30,
		1.0,
	}
}

quake_map_parse_quoted :: proc(source: string, start: int) -> (value: string, next: int, ok: bool) {
	if start >= len(source) || source[start] != '"' {
		return "", start, false
	}

	begin := start + 1
	i := begin
	for i < len(source) {
		if source[i] == '"' {
			return source[begin:i], i + 1, true
		}
		i += 1
	}
	return "", i, false
}

quake_map_skip_ignored :: proc(source: string, start: int) -> int {
	i := start
	for i < len(source) {
		if quake_map_is_space(source[i]) {
			i += 1
			continue
		}
		if i + 1 < len(source) && source[i] == '/' && source[i + 1] == '/' {
			i = quake_map_skip_line(source, i)
			continue
		}
		break
	}
	return i
}

quake_map_skip_line :: proc(source: string, start: int) -> int {
	i := start
	for i < len(source) && source[i] != '\n' {
		i += 1
	}
	return i
}

quake_map_skip_inline_space :: proc(source: string, start: int) -> int {
	i := start
	for i < len(source) && (source[i] == ' ' || source[i] == '\t' || source[i] == '\r') {
		i += 1
	}
	return i
}

quake_map_is_space :: proc(c: u8) -> bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n'
}

#+test
package game

import "core:math"
import "core:strings"
import "core:testing"

TEST_EPSILON :: f32(0.0001)

@(test)
test_quake_map_origin_to_world :: proc(t: ^testing.T) {
	position, ok := quake_map_parse_origin("64 128 256")
	testing.expect(t, ok, "origin should parse")
	testing.expect(t, test_vec3_near(position, {128 * QU_TO_M, 256 * QU_TO_M, 64 * QU_TO_M}), "origin should convert from Quake axes to world axes")
}

@(test)
test_quake_map_origin_rejects_bad_field_count :: proc(t: ^testing.T) {
	_, ok := quake_map_parse_origin("1 2")
	testing.expect(t, !ok, "missing origin component should fail")

	_, ok = quake_map_parse_origin("1 2 3 4")
	testing.expect(t, !ok, "extra origin component should fail")
}

@(test)
test_quake_map_find_player_spawn :: proc(t: ^testing.T) {
	qmap := test_quake_map_from_source(`
{
"classname" "worldspawn"
}
{
"classname" "player"
"origin" "10 20 30"
"angle" "90"
}
`)
	defer quake_map_destroy(&qmap)

	spawn, ok := quake_map_find_player_spawn(&qmap)
	testing.expect(t, ok, "player spawn should be found")
	testing.expect(t, test_vec3_near(spawn.position, {20 * QU_TO_M, 30 * QU_TO_M, 10 * QU_TO_M}), "player spawn position should use world axes")
	testing.expectf(t, test_f32_near(spawn.yaw, math.to_radians(f32(270))), "expected yaw 270 degrees, got %v", spawn.yaw)
}

@(test)
test_quake_map_parse_entities_counts_brushes_and_faces :: proc(t: ^testing.T) {
	qmap := test_quake_map_from_source(`
{
"classname" "worldspawn"
{
( 0 0 0 ) ( 0 0 64 ) ( 0 64 64 ) material 0 0 0 1 1
( 0 0 0 ) ( 64 0 0 ) ( 64 0 64 ) material 0 0 0 1 1
}
}
`)
	defer quake_map_destroy(&qmap)

	brush_count, face_count := quake_map_count_brush_data(&qmap)
	testing.expect_value(t, len(qmap.entities), 1)
	testing.expect_value(t, brush_count, 1)
	testing.expect_value(t, face_count, 2)
}

@(test)
test_quake_map_parse_face_builds_world_plane :: proc(t: ^testing.T) {
	face, ok := quake_map_parse_face("( 0 0 0 ) ( 0 0 64 ) ( 0 64 64 ) green [ 1 0 0 0 ] [ 0 -1 0 0 ] 0 1 1", 0)
	testing.expect(t, ok, "face should parse")
	testing.expect(t, test_vec3_near(face.points[0], {0, 0, 0}), "first point should convert to world axes")
	testing.expect(t, test_vec3_near(face.points[1], {0, 64 * QU_TO_M, 0}), "second point should convert to world axes")
	testing.expect(t, test_vec3_near(face.points[2], {64 * QU_TO_M, 64 * QU_TO_M, 0}), "third point should convert to world axes")
	testing.expect(t, test_vec3_near(face.normal, {0, 0, 1}), "normal should match id/QBSP winding convention")
	testing.expect(t, test_f32_near(face.d, 0), "origin plane should have zero distance")
	testing.expect_value(t, face.material, "green")
	testing.expect_value(t, face.face_suffix, "[ 1 0 0 0 ] [ 0 -1 0 0 ] 0 1 1")
	testing.expect_value(t, face.material_start, 33)
	testing.expect_value(t, face.material_end, 38)
}

@(test)
test_quake_map_face_color_uses_material_color_name :: proc(t: ^testing.T) {
	face, ok := quake_map_parse_face("( 0 0 0 ) ( 0 0 64 ) ( 0 64 64 ) green", 0)
	testing.expect(t, ok, "face should parse")

	color := quake_map_face_color(&face)
	testing.expect(t, test_f32_near(color[0], f32(0.16)), "green material red channel should match palette")
	testing.expect(t, test_f32_near(color[1], f32(0.66)), "green material green channel should match palette")
	testing.expect(t, test_f32_near(color[2], f32(0.22)), "green material blue channel should match palette")
	testing.expect(t, test_f32_near(color[3], f32(1.0)), "material alpha should be opaque")
}

@(test)
test_quake_map_recolor_by_face_normal_assigns_dominant_axis_materials :: proc(t: ^testing.T) {
	face_up := MapFace{normal = {0, 1, 0}, material = "green"}
	face_down := MapFace{normal = {0, -1, 0}, material = "green"}
	face_x := MapFace{normal = {1, 0, 0}, material = "green"}
	face_z := MapFace{normal = {0, 0, -1}, material = "green"}

	testing.expect_value(t, quake_map_material_for_normal(face_up.normal), "green")
	testing.expect_value(t, quake_map_material_for_normal(face_down.normal), "purple")
	testing.expect_value(t, quake_map_material_for_normal(face_x.normal), "red")
	testing.expect_value(t, quake_map_material_for_normal(face_z.normal), "cyan")
}

@(test)
test_quake_map_write_recolored_source_preserves_raw_map_text :: proc(t: ^testing.T) {
	source := `// Game: ArcticChar
// Format: Valve 220
{
"classname" "worldspawn"
{
( 0 0 0 ) ( 0 0 64 ) ( 0 64 64 ) material_with_a_long_name [ 1 0 0 0 ] [ 0 -1 0 0 ] 0 1 1
( -2765.8068218088447 1651.0782104881646 -336 ) ( -2765.8068218088447 1651.0782104881646 -335 ) ( -2765.3068218088442 1650.2121850843807 -336 ) oldtexture [ 0 1.0000000000000002 0 0 ] [ 0 0 -1.0000000000000002 32 ] 0 1 1
}
}
`
	qmap := test_quake_map_from_source(source)
	defer quake_map_destroy(&qmap)
	quake_map_recolor_by_face_normal(&qmap)
	written := quake_map_write_recolored_source(&qmap)
	defer delete(transmute([]byte)written)

	testing.expect(t, strings.contains(written, "// Game: ArcticChar\n// Format: Valve 220"), "writer should preserve header comments")
	testing.expect(t, strings.contains(written, "-2765.8068218088447 1651.0782104881646 -336"), "writer should preserve coordinate precision")
	testing.expect(t, strings.contains(written, "[ 0 1.0000000000000002 0 0 ] [ 0 0 -1.0000000000000002 32 ] 0 1 1"), "writer should preserve face suffix text")
	testing.expect(t, !strings.contains(written, "material_with_a_long_name"), "writer should replace first material token")
	testing.expect(t, !strings.contains(written, "oldtexture"), "writer should replace second material token")

	parsed := test_quake_map_from_source(written)
	defer quake_map_destroy(&parsed)
	quake_map_recolor_by_face_normal(&parsed)
	written_again := quake_map_write_recolored_source(&parsed)
	defer delete(transmute([]byte)written_again)

	testing.expect_value(t, written_again, written)
}

@(test)
test_quake_map_parse_face_rejects_malformed_points :: proc(t: ^testing.T) {
	_, ok := quake_map_parse_face("( 0 0 0 ) ( 0 0 64 )", 0)
	testing.expect(t, !ok, "missing third point should fail")
}

@(test)
test_game_launch_config_map_qpath :: proc(t: ^testing.T) {
	qpath := game_launch_config_map_qpath({map_name = "test"})
	defer delete(qpath)
	testing.expect_value(t, qpath, "maps/test.map")
}

test_quake_map_from_source :: proc(source: string) -> QuakeMap {
	qmap := QuakeMap {
		allocator = context.allocator,
		source = make([]byte, len(source)),
		entities = make([dynamic]MapEntity, 0, 8),
	}
	copy(qmap.source, transmute([]byte)source)
	quake_map_parse_entities(&qmap)
	return qmap
}

test_f32_near :: proc(a, b: f32) -> bool {
	return math.abs(a - b) <= TEST_EPSILON
}

test_vec3_near :: proc(a, b: Vec3) -> bool {
	return test_f32_near(a.x, b.x) && test_f32_near(a.y, b.y) && test_f32_near(a.z, b.z)
}

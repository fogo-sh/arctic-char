#+test
package game

import "core:math"
import "core:testing"
import engine "../engine"

@(test)
test_scene_entity_f32_uses_default_for_missing_or_invalid_values :: proc(t: ^testing.T) {
	entity := test_map_entity({
		{key = "valid", value = "2.5"},
		{key = "invalid", value = "fish"},
	})
	defer test_map_entity_destroy(&entity)

	testing.expect_value(t, scene_entity_f32(&entity, "valid", 1), f32(2.5))
	testing.expect_value(t, scene_entity_f32(&entity, "missing", 7), f32(7))
	testing.expect_value(t, scene_entity_f32(&entity, "invalid", 9), f32(9))
}

@(test)
test_scene_entity_angle_matches_map_yaw_convention :: proc(t: ^testing.T) {
	entity := test_map_entity({{key = "angle", value = "90"}})
	defer test_map_entity_destroy(&entity)

	yaw := scene_entity_angle(&entity, "angle", 0)
	testing.expectf(t, test_f32_near(yaw, math.to_radians(f32(270))), "expected yaw 270 degrees, got %v", yaw)
}

@(test)
test_scene_touch_player_uses_box3d_trigger_shape :: proc(t: ^testing.T) {
	scene := Scene{
		physics = engine.physics_create(),
		objects = make([dynamic]Object, 0, 4),
		next_object_id = 1,
	}
	defer {
		delete(scene.objects)
		engine.physics_destroy(&scene.physics)
	}

	target_position := Vec3{5, 6, 7}
	target_yaw := f32(1.25)
	trigger_id := scene_add_object(
		&scene,
		Object{
			name = "trigger_teleport",
			kind = .Trigger,
			transform = {position = {0, PLAYER_SPEC.capsule_center_y, 0}},
			touch = {
				kind = .TriggerTeleport,
				target_position = target_position,
				target_yaw = target_yaw,
			},
		},
	)
	trigger := scene_object(&scene, trigger_id)
	body, shape := scene_physics_create_trigger_body(&scene, trigger.transform.position, {1, 1, 1})
	trigger.physics = {enabled = true, body = body, shape = shape}

	player := player_create({0, 0, 0})
	scene_touch_player(&scene, &player, {new_position = player.position})

	testing.expect_value(t, player.position, target_position)
	testing.expect_value(t, player.yaw, target_yaw)
}

@(test)
test_trigger_brush_bounds_use_clipped_vertices_not_plane_points :: proc(t: ^testing.T) {
	qmap := test_quake_map_from_source(`
{
"classname" "worldspawn"
}
{
"classname" "trigger_teleport"
{
( -7376 -880 -16 ) ( -7376 -879 -16 ) ( -7376 -880 -15 ) green [ 0 -1 0 0 ] [ 0 0 -1 0 ] 270 1 1
( -4576 -3344 -16 ) ( -4576 -3344 -15 ) ( -4575 -3344 -16 ) green [ 1 0 0 0 ] [ 0 0 -1 0 ] 270 1 1
( -4576 -880 -1248 ) ( -4575 -880 -1248 ) ( -4576 -879 -1248 ) green [ -1 0 0 0 ] [ 0 -1 0 0 ] 270 1 1
( 1328 4576 -960 ) ( 1328 4577 -960 ) ( 1329 4576 -960 ) green [ 1 0 0 0 ] [ 0 -1 0 0 ] 270 1 1
( 1328 6224 16 ) ( 1329 6224 16 ) ( 1328 6224 17 ) green [ -1 0 0 ] [ 0 0 -1 0 ] 270 1 1
( 4144 4576 16 ) ( 4144 4576 17 ) ( 4144 4577 16 ) green [ 0 1 0 0 ] [ 0 0 -1 0 ] 270 1 1
}
}
`)
	defer quake_map_destroy(&qmap)

	entity := &qmap.entities[1]
	bounds_min, bounds_max, ok := map_entity_brush_bounds(entity, map_mesh_base_winding_size(&qmap))

	testing.expect(t, ok, "trigger bounds should be computed")
	testing.expectf(t, bounds_max.y < f32(0.5), "trigger top should remain below spawn height, got %v", bounds_max.y)
}

test_map_entity :: proc(properties: []MapProperty) -> MapEntity {
	entity := MapEntity {
		properties = make([dynamic]MapProperty, 0, len(properties)),
		brushes = make([dynamic]MapBrush, 0),
	}
	for property in properties {
		append(&entity.properties, property)
	}
	return entity
}

test_map_entity_destroy :: proc(entity: ^MapEntity) {
	quake_map_destroy_entity(entity)
}

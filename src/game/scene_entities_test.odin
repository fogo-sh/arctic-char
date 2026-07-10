#+test
package game

import "core:math"
import "core:testing"

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

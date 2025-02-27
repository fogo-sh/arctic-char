package main

import "core:math/linalg"

entities: [dynamic]Entity

Entity :: struct {
	model_info:  ^ModelInfo,
	local_model: matrix[4, 4]f32,
}

entity_create :: proc(
	model: Model,
	position: Vec3 = {0, 0, 0},
	scale: Vec3 = {1, 1, 1},
) -> Entity {
	scale_matrix := linalg.matrix4_scale_f32(scale)
	translation_matrix := linalg.matrix4_translate_f32(position)
	return Entity {
		model_info = &model_info_lookup[model],
		local_model = linalg.matrix_mul(translation_matrix, scale_matrix),
	}
}

entity_create_test_entities :: proc(offset: Vec3 = {0, 0, 0}) {
	for i in 0 ..< 20 {
		pos := Vec3{f32(i * 5), 0, 0} + offset
		entity := entity_create(.Suzanne, pos)
		append(&entities, entity)
	}

	map_scale := Vec3{1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0}
	entity := entity_create(.Map, {0, 0, 0}, map_scale)
	append(&entities, entity)
}

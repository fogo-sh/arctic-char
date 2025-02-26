package main

import "core:math/linalg"

entities: [dynamic]Entity

Entity :: struct {
	model_info:  ^ModelInfo,
	local_model: matrix[4, 4]f32,
}

entity_create :: proc(model: Model, position: Vec3 = {0, 0, 0}) -> Entity {
	return Entity {
		model_info = &model_info_lookup[model],
		local_model = linalg.matrix4_translate_f32(position),
	}
}

entity_create_test_entities :: proc(offset: Vec3 = {0, 0, 0}) {
	for i in 0 ..< 20 {
		pos := Vec3{f32(i * 5), 0, 0} + offset
		entity := entity_create(.Suzanne, pos)
		append(&entities, entity)
	}
}

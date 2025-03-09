package main

import "core:log"
import "core:math/linalg"
import "vendor:box2d"

entities: [dynamic]Entity

Entity :: struct {
	model_info:  ^ModelInfo,
	local_model: matrix[4, 4]f32,
	body:        box2d.BodyId,
	has_body:    bool,
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
		pos := Vec3{f32(i * 5), 10, 0} + offset
		suzanne_entity := entity_create(.Suzanne, pos)

		bd := box2d.DefaultBodyDef()
		bd.isEnabled = true
		bd.type = .dynamicBody
		bd.position = box2d.Vec2{f32(pos.x), f32(pos.y)}
		body := box2d.CreateBody(world_id, bd)

		suzanne_entity.body = body
		suzanne_entity.has_body = true

		sd := box2d.DefaultShapeDef()

		shape := box2d.CreateCircleShape(body, sd, box2d.Circle{center = {0, 0}, radius = 1})

		append(&entities, suzanne_entity)
	}

	map_entity := entity_create(.Map, {0, 0, 0}, {1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0})
	append(&entities, map_entity)

	collision_entity := entity_create(.Collision, {0, 0, 0}, {2.0, 2.0, 2.0})
	append(&entities, collision_entity)
}

entity_update :: proc(entity: ^Entity) {
	if entity.has_body {
		position := box2d.Body_GetPosition(entity.body)

		scale := Vec3 {
			linalg.length(entity.local_model[0].xyz),
			linalg.length(entity.local_model[1].xyz),
			linalg.length(entity.local_model[2].xyz),
		}

		scale_matrix := linalg.matrix4_scale_f32(scale)
		translation_matrix := linalg.matrix4_translate_f32({position.x, position.y, 0})

		entity.local_model = linalg.matrix_mul(translation_matrix, scale_matrix)
	}
}

entity_update_all :: proc() {
	for i := 0; i < len(entities); i += 1 {
		entity_update(&entities[i])
	}
}

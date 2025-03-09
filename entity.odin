package main

import "core:log"
import "core:math/linalg"
import "vendor:box2d"

entities: [dynamic]Entity
player: Entity

Entity :: struct {
	model_info:  ^ModelInfo,
	local_model: matrix[4, 4]f32,
	body:        box2d.BodyId,
	shape:       box2d.ShapeId,
	has_body:    bool,
	scale:       Vec3,
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
		scale = scale,
	}
}

entity_setup_physics :: proc(entity: ^Entity, pos: Vec3) {
	bd := box2d.DefaultBodyDef()
	bd.isEnabled = true
	bd.type = .dynamicBody
	bd.position = box2d.Vec2{f32(pos.x), f32(pos.y)}
	body := box2d.CreateBody(world_id, bd)

	entity.body = body
	entity.has_body = true

	sd := box2d.DefaultShapeDef()

	shape := box2d.CreateCircleShape(body, sd, box2d.Circle{center = {0, 0}, radius = 1})
	entity.shape = shape
}

entity_create_test_entities :: proc(offset: Vec3 = {0, 0, 0}) {
	for i in 0 ..< 20 {
		pos := Vec3{f32(i * 5), 10, 0} + offset
		suzanne_entity := entity_create(.Suzanne, pos)
		entity_setup_physics(&suzanne_entity, pos)
		append(&entities, suzanne_entity)
	}

	map_entity := entity_create(.Map, {0, 0, 0}, {1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0})
	append(&entities, map_entity)

	if show_collision_debug {
		collision_entity := entity_create(.Collision, {0, 0, 0}, {2.0, 2.0, 2.0})
		append(&entities, collision_entity)
	}

	player = entity_create_player()
	append(&entities, player)
}

entity_create_player :: proc() -> Entity {
	player_entity := entity_create(.Fish, {0, 0, 0}, {1.0, 1.0, 1.0})
	entity_setup_physics(&player_entity, {0, 0, 0})
	return player_entity
}

entity_update :: proc(entity: ^Entity) {
	if entity.has_body {
		position := box2d.Body_GetPosition(entity.body)

		if position.y < water_height {
			box2d.Body_SetGravityScale(entity.body, 0)
		} else {
			box2d.Body_SetGravityScale(entity.body, 1)
		}

		scale_matrix := linalg.matrix4_scale_f32(entity.scale)
		translation_matrix := linalg.matrix4_translate_f32({position.x, position.y, 0})

		entity.local_model = linalg.matrix_mul(translation_matrix, scale_matrix)
	}
}

entity_player_collide :: proc(entity: ^Entity) {
	player.scale *= 1.1

	position := box2d.Body_GetPosition(player.body)
	scale_matrix := linalg.matrix4_scale_f32(player.scale)
	translation_matrix := linalg.matrix4_translate_f32({position.x, position.y, 0})
	player.local_model = linalg.matrix_mul(translation_matrix, scale_matrix)

	log.debugf("Player scaled up to: %s", player.scale.x)
}

entity_update_all :: proc() {
	for i := 0; i < len(entities); i += 1 {
		entity_update(&entities[i])
	}
}

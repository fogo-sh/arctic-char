package game

import "core:log"
import "core:c"
import "core:math/linalg"
import "core:strconv"
import b3 "vendor:box3d"

scene_spawn_level_entities :: proc(scene: ^Scene, qmap: ^QuakeMap) {
	for &entity in qmap.entities {
		classname, has_classname := map_entity_property(&entity, "classname")
		if !has_classname {
			continue
		}
		def, known := entity_def_find(classname)
		if !known {
			log.warnf("Skipping unknown map entity classname: %s", classname)
			continue
		}

		switch def.runtime_kind {
		case .Ignore:
			continue
		case .PropSuzanne:
			scene_spawn_entity_prop_suzanne(scene, &entity)
		case .SpawnerSuzanne:
			scene_spawn_entity_spawner_suzanne(scene, &entity)
		case .TriggerTeleport:
			scene_spawn_entity_trigger_teleport(scene, qmap, &entity)
		}
	}
}

scene_clear_level_entities :: proc(scene: ^Scene) {
	for &object in scene.objects {
		if object.physics.enabled && b3.IS_NON_NULL(object.physics.body) {
			b3.DestroyBody(object.physics.body)
		}
	}
	clear(&scene.objects)
	scene_create_map(scene)
}

scene_spawn_entity_prop_suzanne :: proc(scene: ^Scene, entity: ^MapEntity) {
	position, ok := scene_entity_origin(entity)
	if !ok {
		log.warn("prop_suzanne missing origin")
		return
	}
	_ = scene_spawn_suzanne(scene, position, scene_entity_prop_authority(entity))
}

scene_entity_prop_authority :: proc(entity: ^MapEntity) -> ReplicatedPropAuthority {
	policy, ok := map_entity_property(entity, "net_policy")
	if ok && policy == "client" {
		return .ClientOnly
	}
	return .ServerAuthoritative
}

scene_spawn_entity_spawner_suzanne :: proc(scene: ^Scene, entity: ^MapEntity) {
	position, ok := scene_entity_origin(entity)
	if !ok {
		log.warn("spawner_suzanne missing origin")
		return
	}
	if len(scene.objects) >= cap(scene.objects) {
		return
	}

	interval := scene_entity_f32(entity, "interval", SPAWN_INTERVAL)
	count := int(scene_entity_f32(entity, "count", 5))
	if interval <= 0 {
		interval = SPAWN_INTERVAL
	}
	if count < 0 {
		count = 0
	}

	scene_add_object(
		scene,
		Object {
			name = "spawner_suzanne",
			kind = .Spawner,
			transform = {position = position},
			render = {visible = false},
			physics = {enabled = false},
			think = {
				kind = .SpawnerSuzanne,
				interval = interval,
				timer = interval,
				max_count = count,
			},
		},
	)
}

scene_spawn_entity_trigger_teleport :: proc(scene: ^Scene, qmap: ^QuakeMap, entity: ^MapEntity) {
	bounds_min, bounds_max, bounds_ok := map_entity_brush_bounds(entity)
	if !bounds_ok {
		log.warn("trigger_teleport missing brush bounds")
		return
	}

	target_position: Vec3
	target_yaw: f32
	target_ok := false
	if target_origin, ok := map_entity_property(entity, "target_origin"); ok {
		target_position, target_ok = quake_map_parse_origin(target_origin)
		target_yaw = scene_entity_angle(entity, "target_angle", 0)
	} else if target, ok := map_entity_property(entity, "target"); ok {
		target_position, target_yaw, target_ok = scene_find_target_spawn(qmap, target)
	}
	if !target_ok {
		log.warn("trigger_teleport missing target")
		return
	}
	if len(scene.objects) >= cap(scene.objects) {
		return
	}

	trigger_id := scene_add_object(
		scene,
		Object {
			name = "trigger_teleport",
			kind = .Trigger,
			transform = {position = (bounds_min + bounds_max) * 0.5},
			render = {visible = false},
			touch = {
				kind = .TriggerTeleport,
				target_position = target_position,
				target_yaw = target_yaw,
			},
		},
	)
	trigger := scene_object(scene, trigger_id)
	assert(trigger != nil)
	half_extents := (bounds_max - bounds_min) * 0.5
	body, shape := scene_physics_create_trigger_body(scene, trigger.transform.position, half_extents)
	trigger.physics = {
		enabled = true,
		body = body,
		shape = shape,
	}
}

scene_run_think :: proc(scene: ^Scene, step_time: f32) {
	for &object in scene.objects {
		switch object.think.kind {
		case .SpawnerSuzanne:
			scene_think_spawner_suzanne(scene, &object, step_time)
		case .None:
		}
	}
}

scene_think_spawner_suzanne :: proc(scene: ^Scene, object: ^Object, step_time: f32) {
	if object.think.spawned_count >= object.think.max_count {
		return
	}

	object.think.timer += step_time
	for object.think.spawned_count < object.think.max_count && object.think.timer >= object.think.interval {
		if !scene_spawn_suzanne(scene, object.transform.position) {
			return
		}
		object.think.spawned_count += 1
		object.think.timer -= object.think.interval
	}
}

scene_touch_player :: proc(scene: ^Scene, player: ^PlayerController, move: PlayerMoveResult) {
	_ = move.old_position
	query := SceneTouchQuery{scene = scene}
	mover := player_mover_make_capsule(move.new_position, PLAYER_SPEC)
	points := [?]Vec3{Vec3(mover.center1), Vec3(mover.center2)}
	proxy := b3.ShapeProxy{points = raw_data(points[:]), count = c.int(len(points)), radius = mover.radius}
	_ = b3.World_OverlapShape(scene.physics.id, {0, 0, 0}, proxy, physics_player_trigger_query_filter(), scene_touch_player_overlap, &query)
	if query.trigger != nil {
		player_teleport(player, query.trigger.touch.target_position, query.trigger.touch.target_yaw)
	}
}

SceneTouchQuery :: struct {
	scene:   ^Scene,
	trigger: ^Object,
}

scene_touch_player_overlap :: proc "c" (shape: b3.ShapeId, ctx: rawptr) -> bool {
	query := cast(^SceneTouchQuery)ctx
	for &object in query.scene.objects {
		if object.physics.shape == shape && object.touch.kind == .TriggerTeleport {
			query.trigger = &object
			return false
		}
	}
	return true
}

scene_find_target_spawn :: proc(qmap: ^QuakeMap, target: string) -> (position: Vec3, yaw: f32, ok: bool) {
	for &entity in qmap.entities {
		targetname, has_targetname := map_entity_property(&entity, "targetname")
		if !has_targetname || targetname != target {
			continue
		}
		position, ok = scene_entity_origin(&entity)
		if !ok {
			return {}, 0, false
		}
		yaw = scene_entity_angle(&entity, "angle", 0)
		return position, yaw, true
	}
	return {}, 0, false
}

scene_entity_origin :: proc(entity: ^MapEntity) -> (position: Vec3, ok: bool) {
	origin, has_origin := map_entity_property(entity, "origin")
	if !has_origin {
		return {}, false
	}
	return quake_map_parse_origin(origin)
}

scene_entity_f32 :: proc(entity: ^MapEntity, key: string, default_value: f32) -> f32 {
	text, ok := map_entity_property(entity, key)
	if !ok {
		return default_value
	}
	value, parse_ok := strconv.parse_f32(text)
	if !parse_ok {
		return default_value
	}
	return value
}

scene_entity_angle :: proc(entity: ^MapEntity, key: string, default_value: f32) -> f32 {
	return linalg.to_radians(scene_entity_f32(entity, key, default_value) + 180)
}

map_entity_brush_bounds :: proc(entity: ^MapEntity) -> (bounds_min: Vec3, bounds_max: Vec3, ok: bool) {
	bounds_min = {max(f32), max(f32), max(f32)}
	bounds_max = {-max(f32), -max(f32), -max(f32)}
	for &brush in entity.brushes {
		for &face in brush.faces {
			for point in face.points {
				bounds_min.x = min(bounds_min.x, point.x)
				bounds_min.y = min(bounds_min.y, point.y)
				bounds_min.z = min(bounds_min.z, point.z)
				bounds_max.x = max(bounds_max.x, point.x)
				bounds_max.y = max(bounds_max.y, point.y)
				bounds_max.z = max(bounds_max.z, point.z)
				ok = true
			}
		}
	}
	return
}

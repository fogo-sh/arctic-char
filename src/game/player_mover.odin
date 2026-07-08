package game

import "core:c"
import "core:math/linalg"
import b3 "vendor:box3d"

PLAYER_MOVER_ITERATIONS :: 4
PLAYER_MAX_COLLISION_PLANES :: 8

PlayerMoverContext :: struct {
	planes: [PLAYER_MAX_COLLISION_PLANES]b3.CollisionPlane,
	count:  int,
}

player_mover_move :: proc(
	physics: ^PhysicsWorld,
	spec: PlayerSpec,
	center: Vec3,
	displacement: Vec3,
	velocity: Vec3,
	snap_to_ground: bool,
) -> (
	new_center: Vec3,
	new_velocity: Vec3,
	grounded: bool,
	ground_normal: Vec3,
) {
	origin := b3.Pos{0, 0, 0}
	filter := physics_player_query_filter()
	ground_normal = {0, 1, 0}

	mover := player_mover_make_capsule(center, spec)
	player_mover_nudge(physics, &mover, origin, filter)

	planes: PlayerMoverContext
	if snap_to_ground {
		mover, new_velocity, planes = player_mover_ground_move(physics, mover, origin, filter, displacement, velocity, spec)
	} else {
		player_mover_slide(physics, &mover, origin, filter, displacement, spec.walkable_min_y, &planes)
		new_velocity = player_mover_clip_walls(velocity, &planes, spec.walkable_min_y)
	}

	allow_ground_plant := !snap_to_ground || linalg.length(Vec3{displacement.x, 0, displacement.z}) > 0.000001 || velocity.y < -0.000001
	grounded, ground_normal, mover = player_mover_categorize_ground(physics, mover, spec, velocity.y, allow_ground_plant)
	if grounded && new_velocity.y < 0 {
		new_velocity.y = 0
	}

	new_center = player_mover_origin_from_capsule(mover, spec)
	return
}

player_mover_make_capsule :: proc(origin: Vec3, spec: PlayerSpec) -> b3.Capsule {
	center := origin + Vec3{0, spec.capsule_center_y, 0}
	return b3.Capsule{
		center1 = center - Vec3{0, spec.capsule_half_height, 0},
		center2 = center + Vec3{0, spec.capsule_half_height, 0},
		radius = spec.capsule_radius,
	}
}

player_mover_origin_from_capsule :: proc(mover: b3.Capsule, spec: PlayerSpec) -> Vec3 {
	center := (mover.center1 + mover.center2) * 0.5
	return Vec3(center) - Vec3{0, spec.capsule_center_y, 0}
}

player_mover_nudge :: proc(
	physics: ^PhysicsWorld,
	mover: ^b3.Capsule,
	origin: b3.Pos,
	filter: b3.QueryFilter,
) {
	if !player_mover_overlaps(physics, mover^, origin, filter) {
		return
	}

	base := mover^
	NUDGE :: 1.0 / 8.0 * QU_TO_M
	signs := [?]f32{0, -1, 1}
	for z in signs {
		for x in signs {
			for y in signs {
				candidate := base
				offset := Vec3{x * NUDGE, y * NUDGE, z * NUDGE}
				candidate.center1 += offset
				candidate.center2 += offset
				if !player_mover_overlaps(physics, candidate, origin, filter) {
					mover^ = candidate
					return
				}
			}
		}
	}

	mover^ = base
}

player_mover_overlaps :: proc(physics: ^PhysicsWorld, mover: b3.Capsule, origin: b3.Pos, filter: b3.QueryFilter) -> bool {
	ctx: PlayerMoverContext
	b3.World_CollideMover(physics.id, origin, mover, filter, player_mover_collect_planes, &ctx)
	return ctx.count > 0
}

player_mover_ground_move :: proc(
	physics: ^PhysicsWorld,
	mover: b3.Capsule,
	origin: b3.Pos,
	filter: b3.QueryFilter,
	displacement: Vec3,
	velocity: Vec3,
	spec: PlayerSpec,
) -> (new_mover: b3.Capsule, new_velocity: Vec3, planes: PlayerMoverContext) {
	current := mover
	horizontal := Vec3{displacement.x, 0, displacement.z}
	if linalg.length(horizontal) < 0.000001 {
		return current, Vec3{velocity.x, 0, velocity.z}, planes
	}

	direct_frac := b3.World_CastMover(physics.id, origin, current, horizontal, filter, nil, nil)
	if direct_frac >= 1 {
		current.center1 += horizontal
		current.center2 += horizontal
		player_mover_collide_and_solve(physics, &current, origin, filter, &planes)
		return current, Vec3{velocity.x, 0, velocity.z}, planes
	}

	original := current
	down := current
	down_planes: PlayerMoverContext
	player_mover_slide(physics, &down, origin, filter, horizontal, spec.walkable_min_y, &down_planes)
	down_velocity := player_mover_clip_walls(Vec3{velocity.x, 0, velocity.z}, &down_planes, spec.walkable_min_y)

	up := original
	up_frac := b3.World_CastMover(physics.id, origin, up, {0, spec.step_height, 0}, filter, nil, nil)
	if up_frac > 0 {
		up_step := spec.step_height * up_frac
		up.center1 += {0, up_step, 0}
		up.center2 += {0, up_step, 0}
		up_planes: PlayerMoverContext
		player_mover_slide(physics, &up, origin, filter, horizontal, spec.walkable_min_y, &up_planes)

		down_frac := b3.World_CastMover(physics.id, origin, up, {0, -spec.step_height, 0}, filter, nil, nil)
		up.center1 += {0, -spec.step_height * down_frac, 0}
		up.center2 += {0, -spec.step_height * down_frac, 0}
		up_grounded, _, up := player_mover_categorize_ground(physics, up, spec, velocity.y, true)
		if up_grounded {
			down_dist := player_mover_horizontal_distance_sq(original, down)
			up_dist := player_mover_horizontal_distance_sq(original, up)
			if up_dist >= down_dist {
				return up, Vec3{down_velocity.x, 0, down_velocity.z}, up_planes
			}
		}
	}

	return down, down_velocity, down_planes
}

player_mover_horizontal_distance_sq :: proc(from, to: b3.Capsule) -> f32 {
	delta := Vec3(to.center1 - from.center1)
	return delta.x * delta.x + delta.z * delta.z
}

player_mover_slide :: proc(
	physics: ^PhysicsWorld,
	mover: ^b3.Capsule,
	origin: b3.Pos,
	filter: b3.QueryFilter,
	displacement: Vec3,
	walkable_min_y: f32,
	all_planes: ^PlayerMoverContext,
) {
	delta := displacement
	for _ in 0..<PLAYER_MOVER_ITERATIONS {
		if linalg.length(delta) < 0.000001 {
			break
		}

		frac := b3.World_CastMover(physics.id, origin, mover^, delta, filter, nil, nil)
		safe := delta * frac
		mover.center1 += safe
		mover.center2 += safe

		ctx := player_mover_collide_and_solve(physics, mover, origin, filter, all_planes)
		if frac >= 1 {
			break
		}

		delta = player_mover_clip_walls(delta * (1 - frac), &ctx, walkable_min_y)
	}
}

player_mover_collide_and_solve :: proc(
	physics: ^PhysicsWorld,
	mover: ^b3.Capsule,
	origin: b3.Pos,
	filter: b3.QueryFilter,
	all_planes: ^PlayerMoverContext,
) -> PlayerMoverContext {
	ctx: PlayerMoverContext
	b3.World_CollideMover(physics.id, origin, mover^, filter, player_mover_collect_planes, &ctx)
	if ctx.count > 0 {
		result := b3.SolvePlanes({0, 0, 0}, &ctx.planes[0], c.int(ctx.count))
		mover.center1 += result.delta
		mover.center2 += result.delta
		for i in 0..<ctx.count {
			player_mover_add_plane(all_planes, ctx.planes[i])
		}
	}
	return ctx
}

player_mover_collect_planes :: proc "c" (shape: b3.ShapeId, planes: [^]b3.PlaneResult, count: c.int, ctx: rawptr) -> bool {
	mover_ctx := cast(^PlayerMoverContext)ctx
	for i in 0..<int(count) {
		player_mover_add_plane(
			mover_ctx,
			b3.CollisionPlane{plane = planes[i].plane, pushLimit = max(f32), clipVelocity = true},
		)
	}
	return true
}

player_mover_add_plane :: proc "contextless" (ctx: ^PlayerMoverContext, plane: b3.CollisionPlane) {
	if ctx.count >= PLAYER_MAX_COLLISION_PLANES {
		return
	}
	ctx.planes[ctx.count] = plane
	ctx.count += 1
}

player_mover_clip_walls :: proc(velocity: Vec3, ctx: ^PlayerMoverContext, walkable_min_y: f32) -> Vec3 {
	walls: [PLAYER_MAX_COLLISION_PLANES]b3.CollisionPlane
	wall_count := 0
	for i in 0..<ctx.count {
		if ctx.planes[i].plane.normal.y < walkable_min_y {
			walls[wall_count] = ctx.planes[i]
			wall_count += 1
		}
	}
	if wall_count == 0 {
		return velocity
	}
	return b3.ClipVector(velocity, &walls[0], c.int(wall_count))
}

player_mover_categorize_ground :: proc(
	physics: ^PhysicsWorld,
	mover: b3.Capsule,
	spec: PlayerSpec,
	vertical_velocity: f32,
	allow_plant: bool,
) -> (grounded: bool, normal: Vec3, adjusted_mover: b3.Capsule) {
	adjusted_mover = mover
	normal = {0, 1, 0}
	if vertical_velocity > 180.0 * QU_TO_M {
		return false, normal, adjusted_mover
	}

	filter := physics_player_query_filter()
	origin := b3.Pos{0, 0, 0}
	down := Vec3{0, -1.0 * QU_TO_M, 0}
	frac := b3.World_CastMover(physics.id, origin, adjusted_mover, down, filter, nil, nil)
	if frac >= 1 {
		return false, normal, adjusted_mover
	}

	if allow_plant {
		adjusted_mover.center1 += down * frac
		adjusted_mover.center2 += down * frac
	}
	probe_mover := mover
	probe_mover.center1 += down * frac
	probe_mover.center2 += down * frac
	probe_mover.center1 += down * 0.05
	probe_mover.center2 += down * 0.05
	ctx: PlayerMoverContext
	b3.World_CollideMover(physics.id, origin, probe_mover, filter, player_mover_collect_planes, &ctx)
	best_y := f32(-1)
	for i in 0..<ctx.count {
		plane_normal := Vec3(ctx.planes[i].plane.normal)
		if plane_normal.y > best_y {
			best_y = plane_normal.y
			normal = plane_normal
		}
	}

	if best_y >= spec.walkable_min_y {
		return true, normal, adjusted_mover
	}
	if ctx.count == 0 {
		return true, {0, 1, 0}, adjusted_mover
	}
	return false, {0, 1, 0}, mover
}

package main

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

	mover := b3.Capsule{
		center1 = center - Vec3{0, spec.capsule_half_height, 0},
		center2 = center + Vec3{0, spec.capsule_half_height, 0},
		radius = spec.capsule_radius,
	}

	planes: PlayerMoverContext
	player_mover_slide(physics, &mover, origin, filter, displacement, spec.walkable_min_y, &planes)
	new_velocity = player_mover_clip_walls(velocity, &planes, spec.walkable_min_y)

	if snap_to_ground {
		down_frac := b3.World_CastMover(physics.id, origin, mover, {0, -spec.ground_snap, 0}, filter, nil, nil)
		if down_frac < 1 {
			snap_dist := spec.ground_snap * down_frac
			mover.center1 += {0, -snap_dist, 0}
			mover.center2 += {0, -snap_dist, 0}
		}
	}

	grounded, ground_normal = player_mover_ground_probe(physics, mover, spec)
	if grounded && new_velocity.y < 0 {
		new_velocity.y = 0
	}

	new_center = (mover.center1 + mover.center2) * 0.5
	return
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

player_mover_ground_probe :: proc(physics: ^PhysicsWorld, mover: b3.Capsule, spec: PlayerSpec) -> (grounded: bool, normal: Vec3) {
	base := Vec3(mover.center1)
	result := b3.World_CastRayClosest(
		physics.id,
		b3.Pos(base),
		{0, -(spec.capsule_radius + spec.ground_snap + 0.05), 0},
		physics_player_query_filter(),
	)
	if result.hit && result.normal.y >= spec.walkable_min_y {
		return true, result.normal
	}
	return false, {0, 1, 0}
}

package engine

import "base:runtime"
import "core:c"
import "core:math"
import "core:math/linalg"
import b3 "vendor:box3d"

PhysicsWorld :: struct {
	id: b3.WorldId,
}

PhysicsDebugShapeResource :: struct {
	segments: [dynamic][2]Vec3,
}

PHYSICS_STEP_HZ   :: 128
PHYSICS_STEP_TIME :: f32(1.0 / f32(PHYSICS_STEP_HZ))
PHYSICS_SUBSTEPS :: 4
PHYSICS_DYNAMIC_BODY_CAPACITY :: 4096
PHYSICS_STATIC_BODY_CAPACITY :: 64
PHYSICS_STATIC_SHAPE_CAPACITY :: 512
PHYSICS_CONTACT_CAPACITY :: 160000

physics_create :: proc() -> PhysicsWorld {
	world_def := b3.DefaultWorldDef()
	world_def.gravity = {0, -10, 0}
	world_def.createDebugShape = physics_debug_create_shape
	world_def.destroyDebugShape = physics_debug_destroy_shape
	world_def.capacity = {
		staticShapeCount = PHYSICS_STATIC_SHAPE_CAPACITY,
		dynamicShapeCount = PHYSICS_DYNAMIC_BODY_CAPACITY,
		staticBodyCount = PHYSICS_STATIC_BODY_CAPACITY,
		dynamicBodyCount = PHYSICS_DYNAMIC_BODY_CAPACITY,
		contactCount = PHYSICS_CONTACT_CAPACITY,
	}
	return PhysicsWorld{id = b3.CreateWorld(world_def)}
}

physics_debug_draw_world_lines :: proc(physics: ^PhysicsWorld, lines: ^[dynamic]DebugLine) {
	if !b3.IS_NON_NULL(physics.id) do return
	draw := b3.DefaultDebugDraw()
	draw.DrawShapeFcn = physics_debug_draw_shape
	draw.DrawSegmentFcn = physics_debug_draw_segment
	draw.DrawTransformFcn = physics_debug_draw_transform
	draw.DrawPointFcn = physics_debug_draw_point
	draw.DrawSphereFcn = physics_debug_draw_sphere
	draw.DrawCapsuleFcn = physics_debug_draw_capsule
	draw.DrawBoundsFcn = physics_debug_draw_bounds
	draw.DrawBoxFcn = physics_debug_draw_box
	draw.drawShapes = true
	draw.drawBounds = false
	draw.drawJoints = true
	draw.drawContacts = true
	draw.drawContactNormals = true
	draw.jointScale = 1
	draw.forceScale = 1
	draw.ctx = lines
	bounds := b3.World_GetBounds(physics.id)
	draw.drawingBounds = bounds
	b3.World_Draw(physics.id, &draw, b3.DEFAULT_MASK_BITS)
}

physics_debug_create_shape :: proc "c" (debug_shape: ^b3.DebugShape, ctx: rawptr) -> rawptr {
	context = default_context
	resource := new(PhysicsDebugShapeResource)
	resource.segments = make([dynamic][2]Vec3, 0, 64)
	physics_debug_build_shape_segments(resource, debug_shape)
	return resource
}

physics_debug_destroy_shape :: proc "c" (user_shape: rawptr, ctx: rawptr) {
	context = default_context
	if user_shape != nil {
		resource := cast(^PhysicsDebugShapeResource)user_shape
		delete(resource.segments)
		free(resource)
	}
}

physics_debug_draw_shape :: proc "c" (user_shape: rawptr, transform: b3.WorldTransform, color: b3.HexColor, ctx: rawptr) -> bool {
	context = default_context
	resource := cast(^PhysicsDebugShapeResource)user_shape
	if resource == nil do return true
	line_color := physics_debug_color(color)
	for segment in resource.segments {
		physics_debug_append_line(ctx, physics_debug_transform_point(transform, segment[0]), physics_debug_transform_point(transform, segment[1]), line_color)
	}
	return true
}

physics_debug_build_shape_segments :: proc(resource: ^PhysicsDebugShapeResource, shape: ^b3.DebugShape) {
	if shape == nil do return
	#partial switch shape.type {
	case .sphereShape:
		physics_debug_build_sphere_segments(resource, shape.sphere.center, shape.sphere.radius)
	case .capsuleShape:
		physics_debug_build_capsule_segments(resource, shape.capsule.center1, shape.capsule.center2, shape.capsule.radius)
	case .hullShape:
		physics_debug_build_hull_segments(resource, shape.hull)
	case .meshShape:
		if shape.mesh != nil && shape.mesh.data != nil {
			physics_debug_build_aabb_segments(resource, shape.mesh.data.bounds.lowerBound * shape.mesh.scale, shape.mesh.data.bounds.upperBound * shape.mesh.scale)
		}
	case .heightShape:
		if shape.heightField != nil {
			physics_debug_build_aabb_segments(resource, shape.heightField.aabb.lowerBound, shape.heightField.aabb.upperBound)
		}
	case .compoundShape:
		if shape.compound != nil {
			bounds := b3.DynamicTree_GetRootBounds(shape.compound.tree)
			physics_debug_build_aabb_segments(resource, bounds.lowerBound, bounds.upperBound)
		}
	}
}

physics_debug_draw_segment :: proc "c" (p1, p2: b3.Pos, color: b3.HexColor, ctx: rawptr) {
	context = default_context
	physics_debug_append_line(ctx, Vec3(p1), Vec3(p2), physics_debug_color(color))
}

physics_debug_draw_transform :: proc "c" (transform: b3.WorldTransform, ctx: rawptr) {
	context = default_context
	origin := Vec3(transform.p)
	physics_debug_append_line(ctx, origin, Vec3(transform.p) + b3.RotateVector(transform.q, {0.5, 0, 0}), {1, 0.15, 0.15, 1})
	physics_debug_append_line(ctx, origin, Vec3(transform.p) + b3.RotateVector(transform.q, {0, 0.5, 0}), {0.25, 1, 0.25, 1})
	physics_debug_append_line(ctx, origin, Vec3(transform.p) + b3.RotateVector(transform.q, {0, 0, 0.5}), {0.25, 0.45, 1, 1})
}

physics_debug_draw_point :: proc "c" (p: b3.Pos, size: f32, color: b3.HexColor, ctx: rawptr) {
	context = default_context
	s := max(size * 0.005, 0.025)
	center := Vec3(p)
	line_color := physics_debug_color(color)
	physics_debug_append_line(ctx, center - {s, 0, 0}, center + {s, 0, 0}, line_color)
	physics_debug_append_line(ctx, center - {0, s, 0}, center + {0, s, 0}, line_color)
	physics_debug_append_line(ctx, center - {0, 0, s}, center + {0, 0, s}, line_color)
}

physics_debug_draw_sphere :: proc "c" (p: b3.Pos, radius: f32, color: b3.HexColor, alpha: f32, ctx: rawptr) {
	context = default_context
	center := Vec3(p)
	line_color := physics_debug_color(color, alpha)
	SEGMENTS :: 24
	for i in 0..<SEGMENTS {
		a0 := f32(i) / SEGMENTS * 2 * math.PI
		a1 := f32(i + 1) / SEGMENTS * 2 * math.PI
		physics_debug_append_line(ctx, center + {math.cos(a0) * radius, math.sin(a0) * radius, 0}, center + {math.cos(a1) * radius, math.sin(a1) * radius, 0}, line_color)
		physics_debug_append_line(ctx, center + {math.cos(a0) * radius, 0, math.sin(a0) * radius}, center + {math.cos(a1) * radius, 0, math.sin(a1) * radius}, line_color)
		physics_debug_append_line(ctx, center + {0, math.cos(a0) * radius, math.sin(a0) * radius}, center + {0, math.cos(a1) * radius, math.sin(a1) * radius}, line_color)
	}
}

physics_debug_draw_capsule :: proc "c" (p1, p2: b3.Pos, radius: f32, color: b3.HexColor, alpha: f32, ctx: rawptr) {
	context = default_context
	line_color := physics_debug_color(color, alpha)
	a := Vec3(p1)
	b := Vec3(p2)
	physics_debug_draw_sphere(p1, radius, color, alpha, ctx)
	physics_debug_draw_sphere(p2, radius, color, alpha, ctx)
	physics_debug_append_line(ctx, a + {radius, 0, 0}, b + {radius, 0, 0}, line_color)
	physics_debug_append_line(ctx, a - {radius, 0, 0}, b - {radius, 0, 0}, line_color)
	physics_debug_append_line(ctx, a + {0, 0, radius}, b + {0, 0, radius}, line_color)
	physics_debug_append_line(ctx, a - {0, 0, radius}, b - {0, 0, radius}, line_color)
}

physics_debug_draw_bounds :: proc "c" (aabb: b3.AABB, color: b3.HexColor, ctx: rawptr) {
	context = default_context
	physics_debug_draw_aabb(aabb.lowerBound, aabb.upperBound, physics_debug_color(color), ctx)
}

physics_debug_draw_box :: proc "c" (extents: Vec3, transform: b3.WorldTransform, color: b3.HexColor, ctx: rawptr) {
	context = default_context
	corners := physics_debug_aabb_corners(-extents, extents)
	for &corner in corners {
		corner = physics_debug_transform_point(transform, corner)
	}
	physics_debug_draw_box_corners(corners, physics_debug_color(color), ctx)
}

physics_debug_build_sphere_segments :: proc(resource: ^PhysicsDebugShapeResource, center: Vec3, radius: f32) {
	SEGMENTS :: 24
	for i in 0..<SEGMENTS {
		a0 := f32(i) / SEGMENTS * 2 * math.PI
		a1 := f32(i + 1) / SEGMENTS * 2 * math.PI
		physics_debug_resource_add_segment(resource, center + {math.cos(a0) * radius, math.sin(a0) * radius, 0}, center + {math.cos(a1) * radius, math.sin(a1) * radius, 0})
		physics_debug_resource_add_segment(resource, center + {math.cos(a0) * radius, 0, math.sin(a0) * radius}, center + {math.cos(a1) * radius, 0, math.sin(a1) * radius})
		physics_debug_resource_add_segment(resource, center + {0, math.cos(a0) * radius, math.sin(a0) * radius}, center + {0, math.cos(a1) * radius, math.sin(a1) * radius})
	}
}

physics_debug_build_capsule_segments :: proc(resource: ^PhysicsDebugShapeResource, a, b: Vec3, radius: f32) {
	physics_debug_build_sphere_segments(resource, a, radius)
	physics_debug_build_sphere_segments(resource, b, radius)
	physics_debug_resource_add_segment(resource, a + {radius, 0, 0}, b + {radius, 0, 0})
	physics_debug_resource_add_segment(resource, a - {radius, 0, 0}, b - {radius, 0, 0})
	physics_debug_resource_add_segment(resource, a + {0, 0, radius}, b + {0, 0, radius})
	physics_debug_resource_add_segment(resource, a - {0, 0, radius}, b - {0, 0, radius})
}

physics_debug_build_hull_segments :: proc(resource: ^PhysicsDebugShapeResource, hull: ^b3.HullData) {
	if hull == nil do return
	if hull.pointOffset == 0 || hull.edgeOffset == 0 do return
	points := transmute([^]Vec3)(transmute(uintptr)hull + uintptr(hull.pointOffset))
	edges := transmute([^]b3.HullHalfEdge)(transmute(uintptr)hull + uintptr(hull.edgeOffset))
	for i in 0..<int(hull.edgeCount) {
		twin := int(edges[i].twin)
		if i >= twin do continue
		a := edges[i].origin
		b := edges[twin].origin
		physics_debug_resource_add_segment(resource, points[int(a)], points[int(b)])
	}
}

physics_debug_build_aabb_segments :: proc(resource: ^PhysicsDebugShapeResource, mins, maxs: Vec3) {
	corners := physics_debug_aabb_corners(mins, maxs)
	edges := [?][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}}
	for edge in edges {
		physics_debug_resource_add_segment(resource, corners[edge[0]], corners[edge[1]])
	}
}

physics_debug_resource_add_segment :: proc(resource: ^PhysicsDebugShapeResource, from, to: Vec3) {
	append(&resource.segments, [2]Vec3{from, to})
}

physics_debug_transform_point :: proc(transform: b3.WorldTransform, local: Vec3) -> Vec3 {
	model := linalg.matrix4_from_trs_f32(Vec3(transform.p), transform.q, {1, 1, 1})
	return {
		model[0][0] * local.x + model[1][0] * local.y + model[2][0] * local.z + model[3][0],
		model[0][1] * local.x + model[1][1] * local.y + model[2][1] * local.z + model[3][1],
		model[0][2] * local.x + model[1][2] * local.y + model[2][2] * local.z + model[3][2],
	}
}

physics_debug_draw_aabb :: proc(mins, maxs: Vec3, color: Color, ctx: rawptr) {
	corners := physics_debug_aabb_corners(mins, maxs)
	physics_debug_draw_box_corners(corners, color, ctx)
}

physics_debug_aabb_corners :: proc(mins, maxs: Vec3) -> [8]Vec3 {
	return {
		{mins.x, mins.y, mins.z}, {maxs.x, mins.y, mins.z}, {maxs.x, maxs.y, mins.z}, {mins.x, maxs.y, mins.z},
		{mins.x, mins.y, maxs.z}, {maxs.x, mins.y, maxs.z}, {maxs.x, maxs.y, maxs.z}, {mins.x, maxs.y, maxs.z},
	}
}

physics_debug_draw_box_corners :: proc(corners: [8]Vec3, color: Color, ctx: rawptr) {
	edges := [?][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}}
	for edge in edges {
		physics_debug_append_line(ctx, corners[edge[0]], corners[edge[1]], color)
	}
}

physics_debug_append_line :: proc(ctx: rawptr, from, to: Vec3, color: Color) {
	lines := cast(^[dynamic]DebugLine)ctx
	if lines == nil do return
	append(lines, DebugLine{from = from, to = to, color = color})
}

physics_debug_color :: proc(color: b3.HexColor, alpha: f32 = 1) -> Color {
	packed := u32(color) & 0x00ffffff
	return {
		f32((packed >> 16) & 0xff) / 255,
		f32((packed >> 8) & 0xff) / 255,
		f32(packed & 0xff) / 255,
		alpha,
	}
}

physics_destroy :: proc(physics: ^PhysicsWorld) {
	if b3.IS_NON_NULL(physics.id) {
		b3.DestroyWorld(physics.id)
	}
	physics^ = {}
}

physics_step :: proc(physics: ^PhysicsWorld, step_time := PHYSICS_STEP_TIME) {
	b3.World_Step(physics.id, step_time, PHYSICS_SUBSTEPS)
}

physics_create_static_mesh_data :: proc(mesh: ^CpuMesh) -> ^b3.MeshData {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	assert(len(mesh.vertices) >= 3)
	assert(len(mesh.indices) >= 3 && len(mesh.indices) % 3 == 0)

	positions := make([]b3.Vec3, len(mesh.vertices), context.temp_allocator)
	for vertex, i in mesh.vertices {
		positions[i] = vertex.pos
	}

	indices := make([]i32, len(mesh.indices), context.temp_allocator)
	for index, i in mesh.indices {
		assert(index <= u32(max(i32)))
		indices[i] = i32(index)
	}

	degenerate_indices := make([]c.int, len(mesh.indices) / 3, context.temp_allocator)
	mesh_def := b3.MeshDef{
		vertices = raw_data(positions),
		indices = raw_data(indices),
		vertexCount = c.int(len(positions)),
		triangleCount = c.int(len(indices) / 3),
		weldTolerance = 0.01,
		weldVertices = true,
		identifyEdges = true,
	}
	mesh_data := b3.CreateMesh(mesh_def, raw_data(degenerate_indices), c.int(len(degenerate_indices)))
	assert(mesh_data != nil)
	return mesh_data
}

physics_body_matrix :: proc(body: b3.BodyId) -> matrix[4, 4]f32 {
	transform := b3.Body_GetTransform(body)
	return linalg.matrix4_from_trs_f32(b3.ToVec3(transform.p), transform.q, {1, 1, 1})
}

package engine

MeshHandle :: distinct int

RenderItem :: struct {
	mesh:  MeshHandle,
	model: matrix[4, 4]f32,
}

DebugLine :: struct {
	from:  Vec3,
	to:    Vec3,
	color: Color,
}

RenderPassGlobals :: struct {
	view:        matrix[4, 4]f32,
	proj:        matrix[4, 4]f32,
	environment: RenderEnvironment,
}

RenderEnvironment :: struct {
	fog_color:         [4]f32,
	sky_top_color:     [4]f32,
	sky_horizon_color: [4]f32,
	fog_distances:     [4]f32,
}

RendererStats :: struct {
	draw_count:     int,
	triangle_count: int,
	mesh_count:     int,
	pipeline_count: int,
}

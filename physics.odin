package main

import "core:encoding/json"
import "core:fmt"
import "core:log"
import "core:os"
import "vendor:box2d"
import sdl "vendor:sdl3"

Geometry :: struct {
	vertices: [][2]f32,
	edges:    [][2]int,
}

load_box2d_geometry :: proc(
	world: box2d.WorldId,
	vertices: []VertexData,
	indices: []u16,
) -> (
	body: box2d.BodyId,
) {
	body = box2d.CreateBody(world, box2d.BodyDef{type = .staticBody})

	if len(vertices) == 0 || len(indices) == 0 {
		log.error("ModelData has no vertices or indices")
		return
	}

	vertices := make([]box2d.Vec2, len(vertices))
	defer delete(vertices)

	for v, i in vertices {
		vertices[i] = box2d.Vec2{v[0], v[1]}
	}

	for i := 0; i < len(indices); i += 2 {
		if i + 1 >= len(indices) {
			break
		}

		v1_idx := indices[i]
		v2_idx := indices[i + 1]

		if int(v1_idx) >= len(vertices) || int(v2_idx) >= len(vertices) {
			log.error("Index out of bounds in indices")
			continue
		}

		v1 := vertices[v1_idx]
		v2 := vertices[v2_idx]

		segment := box2d.Segment{v1, v2}
		shape_def := box2d.ShapeDef{}
		shape := box2d.CreateSegmentShape(body, shape_def, segment)
	}

	log.debug("Created Box2D geometry from vertices and indices")
	return body
}

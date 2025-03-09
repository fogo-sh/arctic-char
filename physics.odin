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
	body_def := box2d.DefaultBodyDef()
	body_def.type = .staticBody
	body = box2d.CreateBody(world, body_def)

	if len(vertices) == 0 || len(indices) == 0 {
		log.error("ModelData has no vertices or indices")
		return
	}

	box2d_vertices := make([]box2d.Vec2, len(vertices))
	defer delete(box2d_vertices)

	SCALE :: 2.0

	for i in 0 ..< len(vertices) {
		box2d_vertices[i] = box2d.Vec2{vertices[i].pos[0] * SCALE, vertices[i].pos[1] * SCALE}
	}

	chain_vertices := make([dynamic]box2d.Vec2, 0, len(indices))
	defer delete(chain_vertices)

	for i := 0; i < len(indices); i += 1 {
		idx := indices[i]

		if int(idx) >= len(box2d_vertices) {
			log.error("Index out of bounds in indices")
			continue
		}

		append(&chain_vertices, box2d_vertices[idx])
	}

	if len(chain_vertices) > 1 {
		chain_def := box2d.DefaultChainDef()
		chain_def.points = &chain_vertices[0]
		chain_def.count = i32(len(chain_vertices))
		chain_def.isLoop = false

		chain := box2d.CreateChain(body, chain_def)
		log.debug("Created Box2D chain geometry from vertices and indices")
	} else {
		log.error("Not enough valid vertices to create a chain")
	}

	return body
}

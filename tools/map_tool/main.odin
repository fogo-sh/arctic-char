package main

import "base:runtime"
import "core:fmt"
import "core:os"
import game "../../src/game"

main :: proc() {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	args := os.args[1:]
	if len(args) != 2 || args[0] != "recolor" {
		fmt.eprintln("usage: odin run tools/map_tool -- recolor <map-path>")
		os.exit(2)
	}

	path := args[1]
	data, read_err := os.read_entire_file(path, context.allocator)
	if read_err != nil {
		fmt.eprintln("failed to read map:", read_err)
		os.exit(1)
	}
	defer delete(data)

	qmap := game.QuakeMap{
		allocator = context.allocator,
		source = data,
		entities = make([dynamic]game.MapEntity, 0, 16),
	}
	game.quake_map_parse_entities(&qmap)
	game.quake_map_recolor_by_face_normal(&qmap)
	written := game.quake_map_write_recolored_source(&qmap)
	defer delete(transmute([]byte)written)
	qmap.source = nil
	game.quake_map_destroy(&qmap)

	write_err := os.write_entire_file(path, written)
	if write_err != nil {
		fmt.eprintln("failed to write map:", write_err)
		os.exit(1)
	}
	fmt.println("recolored map:", path)
}

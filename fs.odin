package main

import "base:runtime"
import "core:log"
import "core:os"
import path "core:path/filepath"
import "core:strings"

BASE_GAME_DIR :: "base"

GameFS :: struct {
	allocator:    runtime.Allocator,
	base_dir:     string,
	game:         string,
	search_paths: [dynamic]string,
}

game_fs_create :: proc(base_dir, game: string, allocator := context.allocator) -> GameFS {
	fs := GameFS{
		allocator = allocator,
		base_dir = base_dir,
		game = game,
		search_paths = make([dynamic]string, 0, 2, allocator),
	}

	if game != "" && game != BASE_GAME_DIR {
		assert(game_fs_valid_game_dir(game))
		game_fs_add_directory(&fs, game)
	}
	game_fs_add_directory(&fs, BASE_GAME_DIR)
	return fs
}

game_fs_destroy :: proc(fs: ^GameFS) {
	for search_path in fs.search_paths {
		delete(search_path, fs.allocator)
	}
	delete(fs.search_paths)
	fs^ = {}
}

game_fs_add_directory :: proc(fs: ^GameFS, game_dir: string) {
	root, err := path.join({fs.base_dir, game_dir}, fs.allocator)
	assert(err == nil)
	append(&fs.search_paths, root)
	log.debugf("FS search path: %s", root)
}

game_fs_resolve :: proc(fs: ^GameFS, qpath: string, allocator := context.allocator) -> (resolved: string, ok: bool) {
	assert(game_fs_valid_qpath(qpath))
	for root in fs.search_paths {
		candidate, err := path.join({root, qpath}, allocator)
		assert(err == nil)
		if os.is_file(candidate) {
			return candidate, true
		}
		delete(candidate, allocator)
	}
	return "", false
}

game_fs_read_file :: proc(fs: ^GameFS, qpath: string, allocator := context.allocator) -> (data: []byte, ok: bool) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	resolved, found := game_fs_resolve(fs, qpath, context.temp_allocator)
	if !found {
		return nil, false
	}
	contents, err := os.read_entire_file(resolved, allocator)
	return contents, err == nil
}

game_fs_valid_game_dir :: proc(game: string) -> bool {
	return game != "" && !strings.contains(game, "..") && !strings.contains(game, "/") && !strings.contains(game, "\\") && !strings.contains(game, ":")
}

game_fs_valid_qpath :: proc(qpath: string) -> bool {
	return qpath != "" && qpath[0] != '/' && qpath[0] != '\\' && !strings.contains(qpath, "..") && !strings.contains(qpath, ":")
}

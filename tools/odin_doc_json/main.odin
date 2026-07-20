package main

import "base:runtime"
import json "core:encoding/json"
import doc "core:odin/doc-format"
import "core:fmt"
import "core:os"
import "core:strings"

Config :: struct {
	collections: map[string]Collection `json:"collections"`,
}

Collection :: struct {
	source_url:       string   `json:"source_url"`,
	root_path:        string   `json:"root_path"`,
	include_prefixes: []string `json:"include_prefixes"`,
}

Output :: struct {
	schema_version: string        `json:"schemaVersion"`,
	packages:       []Package_Out `json:"packages"`,
	files:          []File_Out    `json:"files"`,
	entities:       []Entity_Out  `json:"entities"`,
	search:         []Search_Out  `json:"search"`,
}

Package_Out :: struct {
	id:         string `json:"id"`,
	name:       string `json:"name"`,
	path:       string `json:"path"`,
	docs:       string `json:"docs"`,
	source_url: string `json:"sourceUrl"`,
}

File_Out :: struct {
	id:         string `json:"id"`,
	package_id: string `json:"packageId"`,
	path:       string `json:"path"`,
	source_url: string `json:"sourceUrl"`,
}

Position_Out :: struct {
	file_id: string `json:"fileId"`,
	line:    int    `json:"line"`,
	column:  int    `json:"column"`,
}

Entity_Out :: struct {
	id:         string       `json:"id"`,
	package_id: string       `json:"packageId"`,
	name:       string       `json:"name"`,
	kind:       string       `json:"kind"`,
	docs:       string       `json:"docs"`,
	position:   Position_Out `json:"position"`,
	source_url: string       `json:"sourceUrl"`,
}

Search_Out :: struct {
	id:         string `json:"id"`,
	type:       string `json:"type"`,
	title:      string `json:"title"`,
	package_id: string `json:"packageId"`,
	text:       string `json:"text"`,
}

main :: proc() {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	args := os.args[1:]
	if len(args) != 3 {
		fmt.eprintln("usage: odin_doc_json <input.odin-doc> <config.json> <output.json>")
		os.exit(2)
	}

	input_path := args[0]
	config_path := args[1]
	output_path := args[2]

	config_data, config_read_err := os.read_entire_file(config_path, context.allocator)
	if config_read_err != nil {
		fmt.eprintln("failed to read config:", config_read_err)
		os.exit(1)
	}
	defer delete(config_data)

	config: Config
	config_err := json.unmarshal(config_data, &config)
	if config_err != nil {
		fmt.eprintln("failed to parse config:", config_err)
		os.exit(1)
	}

	data, read_err := os.read_entire_file(input_path, context.allocator)
	if read_err != nil {
		fmt.eprintln("failed to read odin-doc:", read_err)
		os.exit(1)
	}
	defer delete(data)

	header, doc_err := doc.read_from_bytes(data)
	if doc_err != .None {
		fmt.eprintln("failed to parse odin-doc:", doc_err)
		os.exit(1)
	}

	files := doc.from_array(&header.base, header.files)
	pkgs := doc.from_array(&header.base, header.pkgs)
	entities := doc.from_array(&header.base, header.entities)

	packages_out := make([dynamic]Package_Out, 0, len(pkgs))
	files_out := make([dynamic]File_Out, 0, len(files))
	entities_out := make([dynamic]Entity_Out, 0, len(entities))
	search_out := make([dynamic]Search_Out, 0, len(entities)+len(pkgs))

	package_ids := make([]string, len(pkgs))
	file_ids := make([]string, len(files))
	file_source_urls := make([]string, len(files))
	defer delete(package_ids)
	defer delete(file_ids)
	defer delete(file_source_urls)

	for pkg, i in pkgs {
		path := clean_doc_path(doc.from_string(&header.base, pkg.fullpath))
		name := doc.from_string(&header.base, pkg.name)
		if i == 0 || path == "" {
			continue
		}

		collection, collection_ok := collection_for_path(config, path)
		if !collection_ok {
			continue
		}
		rel_path := relative_to_collection_root(path, collection.root_path)
		if !collection_includes_path(collection, rel_path) {
			continue
		}
		package_id := rel_path
		if package_id == "" {
			package_id = name
		}
		package_ids[i] = package_id

		docs := strings.trim_space(doc.from_string(&header.base, pkg.docs))
		source_url := source_url_for_path(collection, rel_path, 0)
		append(&packages_out, Package_Out{package_id, name, rel_path, docs, source_url})
		append(&search_out, Search_Out{package_id, "package", name, package_id, docs})
	}

	for file, i in files {
		path := clean_doc_path(doc.from_string(&header.base, file.name))
		if i == 0 || path == "" {
			continue
		}

		collection, collection_ok := collection_for_path(config, path)
		if !collection_ok {
			continue
		}
		rel_path := relative_to_collection_root(path, collection.root_path)
		if !collection_includes_path(collection, rel_path) {
			continue
		}
		package_id := ""
		pkg_index := int(file.pkg)
		if pkg_index >= 0 && pkg_index < len(package_ids) {
			package_id = package_ids[pkg_index]
		}

		file_id := rel_path
		file_ids[i] = file_id
		file_source_urls[i] = source_url_for_path(collection, rel_path, 0)
		append(&files_out, File_Out{file_id, package_id, rel_path, file_source_urls[i]})
	}

	for pkg, pkg_i in pkgs {
		if pkg_i == 0 || package_ids[pkg_i] == "" {
			continue
		}
		package_id := package_ids[pkg_i]
		entries := doc.from_array(&header.base, pkg.entries)
		for entry in entries {
			entity_index := int(entry.entity)
			if entity_index <= 0 || entity_index >= len(entities) {
				continue
			}

			entity := entities[entity_index]
			name := doc.from_string(&header.base, entity.name)
			if name == "" {
				name = doc.from_string(&header.base, entry.name)
			}
			if name == "" {
				continue
			}

			file_id := ""
			file_index := int(entity.pos.file)
			if file_index >= 0 && file_index < len(file_ids) {
				file_id = file_ids[file_index]
			}

			docs := strings.trim_space(doc.from_string(&header.base, entity.docs))
			entity_id := fmt.tprintf("%s#%s", package_id, name)
			position := Position_Out{file_id, int(entity.pos.line), int(entity.pos.column)}
			source_url := ""
			if file_index >= 0 && file_index < len(file_ids) && file_ids[file_index] != "" {
				collection, collection_ok := collection_for_path(config, clean_doc_path(doc.from_string(&header.base, files[file_index].name)))
				if collection_ok {
					source_url = source_url_for_path(collection, file_ids[file_index], int(entity.pos.line))
				}
			}

			append(&entities_out, Entity_Out{entity_id, package_id, name, entity_kind_string(entity.kind), docs, position, source_url})
			append(&search_out, Search_Out{entity_id, "entity", name, package_id, docs})
		}
	}

	output := Output{"1", packages_out[:], files_out[:], entities_out[:], search_out[:]}
	json_data, marshal_err := json.marshal(output, {pretty = true, use_spaces = true})
	if marshal_err != nil {
		fmt.eprintln("failed to marshal json:", marshal_err)
		os.exit(1)
	}
	defer delete(json_data)

	write_err := os.write_entire_file(output_path, json_data)
	if write_err != nil {
		fmt.eprintln("failed to write json:", write_err)
		os.exit(1)
	}
}

collection_for_path :: proc(config: Config, path: string) -> (Collection, bool) {
	best: Collection
	best_len := -1
	clean_path := clean_doc_path(path)

	for _, collection in config.collections {
		root := clean_doc_path(expand_path(collection.root_path))
		if strings.has_prefix(clean_path, root) && len(root) > best_len {
			best = collection
			best.root_path = root
			best_len = len(root)
		}
	}

	if best_len >= 0 {
		best.source_url = strings.trim_suffix(best.source_url, "/")
		return best, true
	}

	return {}, false
}

relative_to_collection_root :: proc(path, root_path: string) -> string {
	p := clean_doc_path(path)
	root := clean_doc_path(root_path)
	if root != "" && strings.has_prefix(p, root) {
		p = strings.trim_prefix(p, root)
		p = strings.trim_prefix(p, "/")
	}
	return p
}

source_url_for_path :: proc(collection: Collection, path: string, line: int) -> string {
	if collection.source_url == "" {
		return ""
	}
	url := strings.trim_suffix(collection.source_url, "/")
	clean_path := strings.trim_prefix(clean_doc_path(path), "/")
	if clean_path != "" {
		url = fmt.tprintf("%s/%s", url, clean_path)
	}
	if line > 0 {
		url = fmt.tprintf("%s#L%d", url, line)
	}
	return url
}

expand_path :: proc(path: string) -> string {
	p := strings.trim_space(path)
	if p == "" {
		return ""
	}

	pwd, pwd_err := os.get_working_directory(context.temp_allocator)
	if pwd_err == nil {
		replaced, _ := strings.replace(p, "$PWD", pwd, 1)
		p = replaced
	}

	abs, abs_err := os.get_absolute_path(p, context.temp_allocator)
	if abs_err == nil {
		return abs
	}
	return p
}

clean_doc_path :: proc(path: string) -> string {
	p := strings.trim_space(path)
	when ODIN_OS == .Windows {
		p, _ = strings.replace_all(p, "\\", "/")
	}
	return strings.trim_suffix(p, "/")
}

entity_kind_string :: proc(kind: doc.Entity_Kind) -> string {
	#partial switch kind {
	case .Constant:
		return "constant"
	case .Variable:
		return "variable"
	case .Type_Name:
		return "type"
	case .Procedure:
		return "procedure"
	case .Proc_Group:
		return "proc_group"
	case .Import_Name:
		return "import"
	case .Library_Name:
		return "library"
	case .Builtin:
		return "builtin"
	}
	return "invalid"
}

collection_includes_path :: proc(collection: Collection, path: string) -> bool {
	prefixes := collection.include_prefixes
	if len(prefixes) == 0 {
		return true
	}
	for prefix in prefixes {
		if strings.has_prefix(path, prefix) {
			return true
		}
	}
	return false
}

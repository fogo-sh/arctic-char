package engine_sokol

import "base:runtime"
import engine "../engine"
import sg "../../vendor/sokol/gfx"
import sglue "../../vendor/sokol/glue"
import slog "../../vendor/sokol/log"

SokolRenderer :: struct {
	allocator: runtime.Allocator,
	shader:    sg.Shader,
	pipeline:  sg.Pipeline,
	meshes:    [dynamic]SokolGpuMesh,
	stats:     engine.RendererStats,
}

SokolGpuMesh :: struct {
	vertex_buffer: sg.Buffer,
	index_buffer:  sg.Buffer,
	index_count:   int,
}

create :: proc(allocator := context.allocator) -> SokolRenderer {
	sg.setup({
		environment = sglue.environment(),
		logger = {func = slog.func},
	})

	renderer := SokolRenderer{
		allocator = allocator,
		meshes = make([dynamic]SokolGpuMesh, 0, 16, allocator),
	}
	renderer.shader = sg.make_shader(cube_shader_desc(sg.query_backend()))
	renderer.pipeline = create_pipeline(renderer.shader)
	renderer_update_stats(&renderer)
	return renderer
}

destroy :: proc(renderer: ^SokolRenderer) {
	for &mesh in renderer.meshes {
		destroy_mesh(&mesh)
	}
	delete(renderer.meshes)
	sg.destroy_pipeline(renderer.pipeline)
	sg.destroy_shader(renderer.shader)
	sg.shutdown()
	renderer^ = {}
}

api :: proc(renderer: ^SokolRenderer) -> engine.RendererApi {
	return {
		data = renderer,
		begin_upload = api_begin_upload,
		upload_mesh = api_upload_mesh,
		end_upload = api_end_upload,
		replace_mesh = api_replace_mesh,
	}
}

api_begin_upload :: proc(data: rawptr) -> rawptr {
	_ = data
	return nil
}

api_upload_mesh :: proc(data: rawptr, upload: rawptr, mesh: ^engine.CpuMesh) -> engine.MeshHandle {
	_ = upload
	renderer := cast(^SokolRenderer)data
	gpu_mesh := upload_mesh_data(mesh)
	handle := engine.MeshHandle(len(renderer.meshes))
	append(&renderer.meshes, gpu_mesh)
	renderer_update_stats(renderer)
	return handle
}

api_end_upload :: proc(data: rawptr, upload: rawptr) {
	_ = data
	_ = upload
}

api_replace_mesh :: proc(data: rawptr, handle: engine.MeshHandle, mesh: ^engine.CpuMesh) {
	renderer := cast(^SokolRenderer)data
	index := int(handle)
	assert(0 <= index && index < len(renderer.meshes))
	new_mesh := upload_mesh_data(mesh)
	destroy_mesh(&renderer.meshes[index])
	renderer.meshes[index] = new_mesh
	renderer_update_stats(renderer)
}

upload_mesh_data :: proc(mesh: ^engine.CpuMesh) -> SokolGpuMesh {
	assert(len(mesh.vertices) > 0)
	assert(len(mesh.indices) > 0)
	return {
		vertex_buffer = sg.make_buffer({
			data = {ptr = raw_data(mesh.vertices), size = uint(size_of(engine.VertexData) * len(mesh.vertices))},
		}),
		index_buffer = sg.make_buffer({
			usage = {index_buffer = true},
			data = {ptr = raw_data(mesh.indices), size = uint(size_of(u32) * len(mesh.indices))},
		}),
		index_count = len(mesh.indices),
	}
}

destroy_mesh :: proc(mesh: ^SokolGpuMesh) {
	if mesh.vertex_buffer.id != 0 do sg.destroy_buffer(mesh.vertex_buffer)
	if mesh.index_buffer.id != 0 do sg.destroy_buffer(mesh.index_buffer)
	mesh^ = {}
}

draw :: proc(renderer: ^SokolRenderer, frame: engine.RenderFrame, viewport: [2]i32) {
	_ = viewport
	renderer.stats.draw_count = 0
	renderer.stats.triangle_count = 0

	pass_action := sg.Pass_Action {
		colors = {
			0 = {load_action = .CLEAR, clear_value = {
				r = frame.globals.environment.fog_color.x,
				g = frame.globals.environment.fog_color.y,
				b = frame.globals.environment.fog_color.z,
				a = frame.globals.environment.fog_color.w,
			}},
		},
		depth = {load_action = .CLEAR, clear_value = 1.0},
	}

	sg.begin_pass({action = pass_action, swapchain = sglue.swapchain()})
	sg.apply_pipeline(renderer.pipeline)
	for item in frame.items {
		mesh := renderer_mesh(renderer, item.mesh)
		bind := sg.Bindings{}
		bind.vertex_buffers[0] = mesh.vertex_buffer
		bind.index_buffer = mesh.index_buffer
		sg.apply_bindings(bind)
		vs_params := Vs_Params{mvp = frame.globals.proj * frame.globals.view * item.model}
		sg.apply_uniforms(UB_vs_params, {ptr = &vs_params, size = size_of(vs_params)})
		sg.draw(0, mesh.index_count, 1)
		renderer.stats.draw_count += 1
		renderer.stats.triangle_count += mesh.index_count / 3
	}
	sg.end_pass()
	sg.commit()
}

create_pipeline :: proc(shader: sg.Shader) -> sg.Pipeline {
	return sg.make_pipeline({
		shader = shader,
		layout = {
			buffers = {
				0 = {stride = size_of(engine.VertexData)},
			},
			attrs = {
				ATTR_cube_position = {format = .FLOAT3, offset = i32(offset_of(engine.VertexData, pos))},
				ATTR_cube_color0 = {format = .FLOAT4, offset = i32(offset_of(engine.VertexData, color))},
			},
		},
		index_type = .UINT32,
		cull_mode = .NONE,
		depth = {
			write_enabled = true,
			compare = .LESS_EQUAL,
		},
	})
}

renderer_mesh :: proc(renderer: ^SokolRenderer, handle: engine.MeshHandle) -> ^SokolGpuMesh {
	index := int(handle)
	assert(0 <= index && index < len(renderer.meshes))
	return &renderer.meshes[index]
}

renderer_update_stats :: proc(renderer: ^SokolRenderer) {
	renderer.stats.mesh_count = len(renderer.meshes)
	renderer.stats.pipeline_count = int(renderer.pipeline.id != 0)
}

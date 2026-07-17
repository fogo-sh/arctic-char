package engine_sokol

import "base:runtime"
import "core:log"
import "core:math"
import "core:strings"
import engine "../engine"
import sg "../../vendor/sokol/gfx"
import sglue "../../vendor/sokol/glue"
import slog "../../vendor/sokol/log"

text_msl_vert_shader_code := #load("../../shaders/msl/text.msl.vert")
text_msl_frag_shader_code := #load("../../shaders/msl/text.msl.frag")
text_hlsl_shader_code := #load("../../shaders/hlsl/text.hlsl")

text_glsl300es_vert_shader_code := `#version 300 es
precision highp float;
precision highp int;

layout(std140) uniform SlugVertexUniforms {
    mat4 mvp;
    vec2 viewport;
};

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 tex;
layout(location = 2) in vec4 jac;
layout(location = 3) in vec4 bnd;
layout(location = 4) in vec4 col;

out vec4 color;
out vec2 texcoord;
flat out vec4 banding;
flat out ivec4 glyph;

vec2 SlugDilate(vec4 pos_value, vec4 tex_value, vec4 jac_value, vec4 m0, vec4 m1, vec4 m3, vec2 dim, out vec2 vpos) {
    vec2 n = normalize(pos_value.zw);
    float s = dot(m3.xy, pos_value.xy) + m3.w;
    float t = dot(m3.xy, n);
    float u = (s * dot(m0.xy, n) - t * (dot(m0.xy, pos_value.xy) + m0.w)) * dim.x;
    float v = (s * dot(m1.xy, n) - t * (dot(m1.xy, pos_value.xy) + m1.w)) * dim.y;
    float s2 = s * s;
    float st = s * t;
    float uv = u * u + v * v;
    vec2 d = pos_value.zw * (s2 * (st + sqrt(uv)) / (uv - st * st));
    vpos = pos_value.xy + d;
    return vec2(tex_value.x + dot(d, jac_value.xy), tex_value.y + dot(d, jac_value.zw));
}

void main() {
    vec2 p;
    vec4 m0 = vec4(mvp[0][0], mvp[1][0], mvp[2][0], mvp[3][0]);
    vec4 m1 = vec4(mvp[0][1], mvp[1][1], mvp[2][1], mvp[3][1]);
    vec4 m2 = vec4(mvp[0][2], mvp[1][2], mvp[2][2], mvp[3][2]);
    vec4 m3 = vec4(mvp[0][3], mvp[1][3], mvp[2][3], mvp[3][3]);
    texcoord = SlugDilate(pos, tex, jac, m0, m1, m3, viewport, p);
    gl_Position = vec4(
        p.x * m0.x + p.y * m0.y + m0.w,
        p.x * m1.x + p.y * m1.y + m1.w,
        p.x * m2.x + p.y * m2.y + m2.w,
        p.x * m3.x + p.y * m3.y + m3.w);
    uvec2 g = floatBitsToUint(tex.zw);
    glyph = ivec4(int(g.x & 65535u), int(g.x >> 16u), int(g.y & 65535u), int(g.y >> 16u));
    banding = bnd;
    color = col;
}
`

text_glsl300es_frag_shader_code := `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

layout(std140) uniform SlugFragmentUniforms {
    float weightBoost;
};

uniform sampler2D curveTexture;
uniform usampler2D bandTexture;

in vec4 color;
in vec2 texcoord;
flat in vec4 banding;
flat in ivec4 glyph;
layout(location = 0) out vec4 frag_color;

const int kLogBandTextureWidth = 12;

uint CalcRootCode(float y1, float y2, float y3) {
    uint i1 = floatBitsToUint(y1) >> 31u;
    uint i2 = floatBitsToUint(y2) >> 30u;
    uint i3 = floatBitsToUint(y3) >> 29u;
    uint shift = (i2 & 2u) | (i1 & ~2u);
    shift = (i3 & 4u) | (shift & ~4u);
    return ((11892u >> shift) & 257u);
}

vec2 SolveHorizPoly(vec4 p12, vec2 p3) {
    vec2 a = p12.xy - p12.zw * 2.0 + p3;
    vec2 b = p12.xy - p12.zw;
    float ra = 1.0 / a.y;
    float rb = 0.5 / b.y;
    float d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    float t1 = (b.y - d) * ra;
    float t2 = (b.y + d) * ra;
    if (abs(a.y) < 1.0 / 65536.0) {
        t1 = p12.y * rb;
        t2 = t1;
    }
    return vec2((a.x * t1 - b.x * 2.0) * t1 + p12.x, (a.x * t2 - b.x * 2.0) * t2 + p12.x);
}

vec2 SolveVertPoly(vec4 p12, vec2 p3) {
    vec2 a = p12.xy - p12.zw * 2.0 + p3;
    vec2 b = p12.xy - p12.zw;
    float ra = 1.0 / a.x;
    float rb = 0.5 / b.x;
    float d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    float t1 = (b.x - d) * ra;
    float t2 = (b.x + d) * ra;
    if (abs(a.x) < 1.0 / 65536.0) {
        t1 = p12.x * rb;
        t2 = t1;
    }
    return vec2((a.y * t1 - b.y * 2.0) * t1 + p12.y, (a.y * t2 - b.y * 2.0) * t2 + p12.y);
}

ivec2 CalcBandLoc(ivec2 glyphLoc, uint offset) {
    ivec2 bandLoc = ivec2(glyphLoc.x + int(offset), glyphLoc.y);
    bandLoc.y += bandLoc.x >> kLogBandTextureWidth;
    bandLoc.x &= (1 << kLogBandTextureWidth) - 1;
    return bandLoc;
}

float CalcCoverage(float xcov, float ycov, float xwgt, float ywgt) {
    float coverage = max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0), min(abs(xcov), abs(ycov)));
    return clamp(coverage, 0.0, 1.0);
}

void main() {
    vec2 renderCoord = texcoord;
    vec4 bandTransform = banding;
    ivec4 glyphData = glyph;
    vec2 emsPerPixel = fwidth(renderCoord);
    vec2 pixelsPerEm = 1.0 / emsPerPixel;
    ivec2 bandMax = glyphData.zw;
    bandMax.y &= 255;
    ivec2 bandIndex = clamp(ivec2(renderCoord * bandTransform.xy + bandTransform.zw), ivec2(0), bandMax);
    ivec2 glyphLoc = glyphData.xy;

    float xcov = 0.0;
    float xwgt = 0.0;
    uvec2 hbandData = texelFetch(bandTexture, ivec2(glyphLoc.x + bandIndex.y, glyphLoc.y), 0).xy;
    ivec2 hbandLoc = CalcBandLoc(glyphLoc, hbandData.y);
    for (int curveIndex = 0; curveIndex < int(hbandData.x); curveIndex++) {
        ivec2 curveLoc = ivec2(texelFetch(bandTexture, ivec2(hbandLoc.x + curveIndex, hbandLoc.y), 0).xy);
        vec4 p12 = texelFetch(curveTexture, curveLoc, 0) - vec4(renderCoord, renderCoord);
        vec2 p3 = texelFetch(curveTexture, ivec2(curveLoc.x + 1, curveLoc.y), 0).xy - renderCoord;
        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) break;
        uint code = CalcRootCode(p12.y, p12.w, p3.y);
        if (code != 0u) {
            vec2 r = SolveHorizPoly(p12, p3) * pixelsPerEm.x;
            if ((code & 1u) != 0u) {
                xcov += clamp(r.x + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code > 1u) {
                xcov -= clamp(r.y + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    float ycov = 0.0;
    float ywgt = 0.0;
    uvec2 vbandData = texelFetch(bandTexture, ivec2(glyphLoc.x + bandMax.y + 1 + bandIndex.x, glyphLoc.y), 0).xy;
    ivec2 vbandLoc = CalcBandLoc(glyphLoc, vbandData.y);
    for (int curveIndex = 0; curveIndex < int(vbandData.x); curveIndex++) {
        ivec2 curveLoc = ivec2(texelFetch(bandTexture, ivec2(vbandLoc.x + curveIndex, vbandLoc.y), 0).xy);
        vec4 p12 = texelFetch(curveTexture, curveLoc, 0) - vec4(renderCoord, renderCoord);
        vec2 p3 = texelFetch(curveTexture, ivec2(curveLoc.x + 1, curveLoc.y), 0).xy - renderCoord;
        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) break;
        uint code = CalcRootCode(p12.x, p12.z, p3.x);
        if (code != 0u) {
            vec2 r = SolveVertPoly(p12, p3) * pixelsPerEm.y;
            if ((code & 1u) != 0u) {
                ycov -= clamp(r.x + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code > 1u) {
                ycov += clamp(r.y + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    float coverage = CalcCoverage(xcov, ycov, xwgt, ywgt);
    if (weightBoost > 0.5) coverage = sqrt(coverage);
    frag_color = vec4(color.rgb, color.a * coverage);
}
`

DEBUG_LINE_CAPACITY :: 32768
UI_VERTEX_CAPACITY :: 8192

SokolTextVertexUniforms :: struct #align(16) {
	mvp:      matrix[4, 4]f32,
	viewport: [2]f32,
	padding:  [2]f32,
}

SokolTextFragmentUniforms :: struct #align(16) {
	weight_boost: f32,
	padding:      [3]f32,
}

SokolTextDrawRange :: struct {
	first_index:    int,
	index_count:    int,
	scissor_active: bool,
	scissor_bounds: [4]f32,
}

SokolTextRenderer :: struct {
	ctx:           engine.Text_Context,
	draw_ranges:   [dynamic]SokolTextDrawRange,
	font_pack:     engine.Text_Texture_Pack,
	shader:        sg.Shader,
	pipeline:      sg.Pipeline,
	vertex_buffer: sg.Buffer,
	index_buffer:  sg.Buffer,
	curve_image:   sg.Image,
	band_image:    sg.Image,
	curve_view:    sg.View,
	band_view:     sg.View,
	curve_sampler: sg.Sampler,
	band_sampler:  sg.Sampler,
	loaded:        bool,
}

SokolRenderer :: struct {
	allocator: runtime.Allocator,
	shader:    sg.Shader,
	pipeline:  sg.Pipeline,
	line_pipeline: sg.Pipeline,
	sky_pipeline:  sg.Pipeline,
	ui_pipeline:   sg.Pipeline,
	sky_vertex_buffer: sg.Buffer,
	debug_vertex_buffer: sg.Buffer,
	ui_vertex_buffer: sg.Buffer,
	debug_vertices: [dynamic]engine.VertexData,
	ui_vertices: [dynamic]engine.VertexData,
	ui_draws: [dynamic]engine.UiGeometryDraw,
	text:      ^SokolTextRenderer,
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
	renderer.pipeline = create_pipeline(renderer.shader, .TRIANGLES, .UINT32, depth_enabled = true, blend_enabled = false)
	renderer.line_pipeline = create_pipeline(renderer.shader, .LINES, .NONE, depth_enabled = false, blend_enabled = false)
	renderer.sky_pipeline = create_pipeline(renderer.shader, .TRIANGLES, .NONE, depth_enabled = false, blend_enabled = false)
	renderer.ui_pipeline = create_pipeline(renderer.shader, .TRIANGLES, .NONE, depth_enabled = false, blend_enabled = true)
	renderer.sky_vertex_buffer = sg.make_buffer({
		size = 3 * size_of(engine.VertexData),
		usage = {vertex_buffer = true, stream_update = true},
	})
	renderer.debug_vertex_buffer = sg.make_buffer({
		size = DEBUG_LINE_CAPACITY * 2 * size_of(engine.VertexData),
		usage = {vertex_buffer = true, stream_update = true},
	})
	renderer.ui_vertex_buffer = sg.make_buffer({
		size = UI_VERTEX_CAPACITY * size_of(engine.VertexData),
		usage = {vertex_buffer = true, stream_update = true},
	})
	renderer.debug_vertices = make([dynamic]engine.VertexData, 0, DEBUG_LINE_CAPACITY * 2, allocator)
	renderer.ui_vertices = make([dynamic]engine.VertexData, 0, UI_VERTEX_CAPACITY, allocator)
	renderer.ui_draws = make([dynamic]engine.UiGeometryDraw, 0, engine.UI_COMMAND_CAPACITY, allocator)
	renderer.text = create_text_renderer()
	renderer_update_stats(&renderer)
	return renderer
}

destroy :: proc(renderer: ^SokolRenderer) {
	for &mesh in renderer.meshes {
		destroy_mesh(&mesh)
	}
	delete(renderer.ui_vertices)
	delete(renderer.debug_vertices)
	delete(renderer.ui_draws)
	delete(renderer.meshes)
	destroy_text_renderer(renderer.text)
	sg.destroy_buffer(renderer.ui_vertex_buffer)
	sg.destroy_buffer(renderer.debug_vertex_buffer)
	sg.destroy_buffer(renderer.sky_vertex_buffer)
	sg.destroy_pipeline(renderer.ui_pipeline)
	sg.destroy_pipeline(renderer.sky_pipeline)
	sg.destroy_pipeline(renderer.line_pipeline)
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

draw :: proc(renderer: ^SokolRenderer, frame: engine.RenderFrame, viewport: [2]i32, ui_commands: []engine.UiCommand = nil, ui_scale: [2]f32 = {1, 1}) {
	renderer.stats.draw_count = 0
	renderer.stats.triangle_count = 0
	prepare_debug_lines(renderer, frame.debug_lines)
	prepare_ui(renderer, ui_commands, viewport, ui_scale)
	prepare_text(renderer, ui_commands, ui_scale)

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
	draw_sky(renderer, frame.globals)
	draw_world(renderer, frame)
	draw_debug_lines(renderer, frame.globals)
	draw_ui(renderer, viewport)
	draw_text(renderer, viewport)
	sg.end_pass()
	sg.commit()
}

draw_world :: proc(renderer: ^SokolRenderer, frame: engine.RenderFrame) {
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
}

draw_sky :: proc(renderer: ^SokolRenderer, globals: engine.RenderPassGlobals) {
	prepare_sky(renderer, globals.environment)
	sg.apply_pipeline(renderer.sky_pipeline)
	bind := sg.Bindings{}
	bind.vertex_buffers[0] = renderer.sky_vertex_buffer
	sg.apply_bindings(bind)
	vs_params := Vs_Params{mvp = matrix4_identity()}
	sg.apply_uniforms(UB_vs_params, {ptr = &vs_params, size = size_of(vs_params)})
	sg.draw(0, 3, 1)
	renderer.stats.draw_count += 1
	renderer.stats.triangle_count += 1
}

draw_debug_lines :: proc(renderer: ^SokolRenderer, globals: engine.RenderPassGlobals) {
	if len(renderer.debug_vertices) == 0 do return
	sg.apply_pipeline(renderer.line_pipeline)
	bind := sg.Bindings{}
	bind.vertex_buffers[0] = renderer.debug_vertex_buffer
	sg.apply_bindings(bind)
	vs_params := Vs_Params{mvp = globals.proj * globals.view}
	sg.apply_uniforms(UB_vs_params, {ptr = &vs_params, size = size_of(vs_params)})
	sg.draw(0, len(renderer.debug_vertices), 1)
	renderer.stats.draw_count += 1
}

draw_ui :: proc(renderer: ^SokolRenderer, viewport: [2]i32) {
	if len(renderer.ui_vertices) == 0 do return
	sg.apply_pipeline(renderer.ui_pipeline)
	bind := sg.Bindings{}
	bind.vertex_buffers[0] = renderer.ui_vertex_buffer
	sg.apply_bindings(bind)
	vs_params := Vs_Params{mvp = ui_ortho(f32(viewport.x), f32(viewport.y))}
	sg.apply_uniforms(UB_vs_params, {ptr = &vs_params, size = size_of(vs_params)})
	apply_full_scissor(viewport)
	for draw in renderer.ui_draws {
		switch draw.kind {
		case .Geometry:
			if draw.count == 0 do continue
			sg.draw(draw.start, draw.count, 1)
			renderer.stats.draw_count += 1
			renderer.stats.triangle_count += draw.count / 3
		case .ScissorStart:
			apply_ui_scissor(draw.bounds, viewport)
		case .ScissorEnd:
			apply_full_scissor(viewport)
		}
	}
	apply_full_scissor(viewport)
}

draw_text :: proc(renderer: ^SokolRenderer, viewport: [2]i32) {
	text := renderer.text
	if text == nil || !text.loaded || text.ctx.quad_count == 0 do return
	sg.apply_pipeline(text.pipeline)
	bind := sg.Bindings{}
	bind.vertex_buffers[0] = text.vertex_buffer
	bind.index_buffer = text.index_buffer
	bind.views[0] = text.curve_view
	bind.views[1] = text.band_view
	bind.samplers[0] = text.curve_sampler
	bind.samplers[1] = text.band_sampler
	sg.apply_bindings(bind)
	vertex_uniforms := SokolTextVertexUniforms{
		mvp = text_ortho(f32(viewport.x), f32(viewport.y)),
		viewport = {f32(viewport.x), f32(viewport.y)},
	}
	fragment_uniforms := SokolTextFragmentUniforms{}
	sg.apply_uniforms(0, {ptr = &vertex_uniforms, size = size_of(vertex_uniforms)})
	sg.apply_uniforms(1, {ptr = &fragment_uniforms, size = size_of(fragment_uniforms)})
	apply_full_scissor(viewport)
	for range in text.draw_ranges {
		if range.scissor_active {
			apply_ui_scissor(range.scissor_bounds, viewport)
		} else {
			apply_full_scissor(viewport)
		}
		sg.draw(range.first_index, range.index_count, 1)
		renderer.stats.draw_count += 1
		renderer.stats.triangle_count += range.index_count / 3
	}
	apply_full_scissor(viewport)
}

create_pipeline :: proc(shader: sg.Shader, primitive_type: sg.Primitive_Type, index_type: sg.Index_Type, depth_enabled, blend_enabled: bool) -> sg.Pipeline {
	blend: sg.Blend_State
	if blend_enabled {
		blend = {
			enabled = true,
			src_factor_rgb = .SRC_ALPHA,
			dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
			op_rgb = .ADD,
			src_factor_alpha = .ONE,
			dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
			op_alpha = .ADD,
		}
	}
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
		index_type = index_type,
		primitive_type = primitive_type,
		cull_mode = .NONE,
		depth = {
			write_enabled = depth_enabled,
			compare = depth_enabled ? sg.Compare_Func.LESS_EQUAL : sg.Compare_Func.ALWAYS,
		},
		colors = {0 = {blend = blend}},
	})
}

prepare_sky :: proc(renderer: ^SokolRenderer, environment: engine.RenderEnvironment) {
	vertices := [?]engine.VertexData{
		{pos = {-1, -1, 0}, color = environment.sky_horizon_color},
		{pos = { 3, -1, 0}, color = environment.sky_horizon_color},
		{pos = {-1,  3, 0}, color = environment.sky_top_color},
	}
	sg.update_buffer(renderer.sky_vertex_buffer, {ptr = &vertices, size = size_of(vertices)})
}

prepare_debug_lines :: proc(renderer: ^SokolRenderer, lines: []engine.DebugLine) {
	clear(&renderer.debug_vertices)
	if len(lines) == 0 do return
	line_count := min(len(lines), DEBUG_LINE_CAPACITY)
	for line in lines[:line_count] {
		append(&renderer.debug_vertices, engine.VertexData{pos = line.from, color = line.color})
		append(&renderer.debug_vertices, engine.VertexData{pos = line.to, color = line.color})
	}
	sg.update_buffer(renderer.debug_vertex_buffer, {ptr = raw_data(renderer.debug_vertices[:]), size = uint(len(renderer.debug_vertices) * size_of(engine.VertexData))})
}

prepare_ui :: proc(renderer: ^SokolRenderer, commands: []engine.UiCommand, viewport: [2]i32, ui_scale: [2]f32) {
	_ = viewport
	clear(&renderer.ui_vertices)
	clear(&renderer.ui_draws)
	if len(commands) == 0 do return
	for command in commands {
		scaled := engine.ui_command_scaled(command, ui_scale)
		#partial switch command.kind {
		case .Rectangle:
			engine.ui_geometry_append_command(&renderer.ui_vertices, &renderer.ui_draws, scaled)
		case .Border:
			engine.ui_geometry_append_command(&renderer.ui_vertices, &renderer.ui_draws, scaled)
		case .Text:
			// Text rendering is a separate Sokol text pipeline; keep UI geometry useful now.
		case .ScissorStart:
			append(&renderer.ui_draws, engine.UiGeometryDraw{kind = .ScissorStart, bounds = scaled.bounds})
		case .ScissorEnd:
			append(&renderer.ui_draws, engine.UiGeometryDraw{kind = .ScissorEnd})
		}
	}
	if len(renderer.ui_vertices) > 0 {
		sg.update_buffer(renderer.ui_vertex_buffer, {ptr = raw_data(renderer.ui_vertices[:]), size = uint(len(renderer.ui_vertices) * size_of(engine.VertexData))})
	}
}

apply_full_scissor :: proc(viewport: [2]i32) {
	sg.apply_scissor_rect(0, 0, viewport.x, viewport.y, true)
}

apply_ui_scissor :: proc(bounds: [4]f32, viewport: [2]i32) {
	x0 := max(i32(math.floor(bounds.x)), 0)
	y0 := max(i32(math.floor(bounds.y)), 0)
	x1 := min(i32(math.ceil(bounds.x + bounds.z)), viewport.x)
	y1 := min(i32(math.ceil(bounds.y + bounds.w)), viewport.y)
	sg.apply_scissor_rect(x0, y0, max(x1 - x0, 0), max(y1 - y0, 0), true)
}

create_text_renderer :: proc() -> ^SokolTextRenderer {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	text := new(SokolTextRenderer)
	text.draw_ranges = make([dynamic]SokolTextDrawRange, 0, engine.UI_COMMAND_CAPACITY)

	text.shader = create_text_shader()
	if text.shader.id == 0 {
		return text
	}
	text.pipeline = create_text_pipeline(text.shader)
	text.vertex_buffer = sg.make_buffer({
		size = engine.TEXT_MAX_GLYPH_VERTICES * size_of(engine.Text_Vertex),
		usage = {vertex_buffer = true, stream_update = true},
	})
	indices := make([]u32, engine.TEXT_MAX_GLYPH_INDICES, context.temp_allocator)
	for quad in 0..<engine.TEXT_MAX_GLYPH_QUADS {
		base_vertex := u32(quad * engine.TEXT_VERTICES_PER_QUAD)
		base_index := quad * engine.TEXT_INDICES_PER_QUAD
		indices[base_index + 0] = base_vertex + 0
		indices[base_index + 1] = base_vertex + 1
		indices[base_index + 2] = base_vertex + 2
		indices[base_index + 3] = base_vertex + 2
		indices[base_index + 4] = base_vertex + 3
		indices[base_index + 5] = base_vertex + 0
	}
	text.index_buffer = sg.make_buffer({
		size = len(indices) * size_of(u32),
		usage = {index_buffer = true},
		data = {ptr = raw_data(indices), size = uint(len(indices) * size_of(u32))},
	})

	font_path := "/System/Library/Fonts/Supplemental/Arial.ttf"
	font, font_ok := engine.text_font_load(font_path)
	if !font_ok {
		log.warnf("Sokol text font unavailable: %s", font_path)
		return text
	}
	engine.text_font_load_ascii(&font)
	engine.text_register_font(&text.ctx, font)
	text.font_pack = engine.text_font_process(&text.ctx.font)
	text.curve_image = create_text_image(.RGBA16F, text.font_pack.curve_width, text.font_pack.curve_height, raw_data(text.font_pack.curve_data), len(text.font_pack.curve_data) * size_of([4]u16))
	text.band_image = create_text_image(.RG16UI, text.font_pack.band_width, text.font_pack.band_height, raw_data(text.font_pack.band_data), len(text.font_pack.band_data) * size_of([2]u16))
	text.curve_view = sg.make_view({texture = {image = text.curve_image}})
	text.band_view = sg.make_view({texture = {image = text.band_image}})
	text.curve_sampler = sg.make_sampler({min_filter = .NEAREST, mag_filter = .NEAREST})
	text.band_sampler = sg.make_sampler({min_filter = .NEAREST, mag_filter = .NEAREST})
	text.loaded = text.shader.id != 0 && text.pipeline.id != 0 && text.vertex_buffer.id != 0 && text.index_buffer.id != 0 && text.curve_view.id != 0 && text.band_view.id != 0 && text.curve_sampler.id != 0 && text.band_sampler.id != 0
	return text
}

destroy_text_renderer :: proc(text: ^SokolTextRenderer) {
	if text == nil do return
	if text.band_view.id != 0 do sg.destroy_view(text.band_view)
	if text.curve_view.id != 0 do sg.destroy_view(text.curve_view)
	if text.band_sampler.id != 0 do sg.destroy_sampler(text.band_sampler)
	if text.curve_sampler.id != 0 do sg.destroy_sampler(text.curve_sampler)
	if text.band_image.id != 0 do sg.destroy_image(text.band_image)
	if text.curve_image.id != 0 do sg.destroy_image(text.curve_image)
	if text.index_buffer.id != 0 do sg.destroy_buffer(text.index_buffer)
	if text.vertex_buffer.id != 0 do sg.destroy_buffer(text.vertex_buffer)
	if text.pipeline.id != 0 do sg.destroy_pipeline(text.pipeline)
	if text.shader.id != 0 do sg.destroy_shader(text.shader)
	delete(text.draw_ranges)
	engine.text_pack_destroy(&text.font_pack)
	engine.text_context_destroy(&text.ctx)
	text^ = {}
	free(text)
}

prepare_text :: proc(renderer: ^SokolRenderer, commands: []engine.UiCommand, ui_scale: [2]f32) {
	text := renderer.text
	if text == nil || !text.loaded do return
	engine.text_begin(&text.ctx)
	clear(&text.draw_ranges)
	font := &text.ctx.font
	scissor_active := false
	scissor_bounds: [4]f32
	for command in commands {
		scaled := engine.ui_command_scaled(command, ui_scale)
		#partial switch command.kind {
		case .ScissorStart:
			scissor_active = true
			scissor_bounds = scaled.bounds
		case .ScissorEnd:
			scissor_active = false
		case .Text:
			if command.text == "" do continue
			start_quad := text.ctx.quad_count
			baseline_y := scaled.bounds.y + font.ascent * scaled.font_size
			engine.text_draw(&text.ctx, command.text, scaled.bounds.x, baseline_y, scaled.font_size, scaled.color)
			quad_count := text.ctx.quad_count - start_quad
			if quad_count > 0 {
				append(&text.draw_ranges, SokolTextDrawRange{
					first_index = int(start_quad * engine.TEXT_INDICES_PER_QUAD),
					index_count = int(quad_count * engine.TEXT_INDICES_PER_QUAD),
					scissor_active = scissor_active,
					scissor_bounds = scissor_bounds,
				})
			}
		case:
		}
	}
	vertex_count := engine.text_vertex_count(&text.ctx)
	if vertex_count == 0 do return
	sg.update_buffer(text.vertex_buffer, {ptr = &text.ctx.vertices[0], size = uint(int(vertex_count) * size_of(engine.Text_Vertex))})
}

create_text_image :: proc(format: sg.Pixel_Format, width, height: u32, data: rawptr, size: int) -> sg.Image {
	return sg.make_image({
		width = i32(width),
		height = i32(height),
		pixel_format = format,
		data = {mip_levels = {0 = {ptr = data, size = uint(size)}}},
	})
}

create_text_shader :: proc() -> sg.Shader {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	backend := sg.query_backend()
	desc := sg.Shader_Desc{}
	desc.label = "sokol_text_shader"
	#partial switch backend {
	case .METAL_MACOS:
		desc.vertex_func = {source = strings.clone_to_cstring(string(text_msl_vert_shader_code), context.temp_allocator), entry = "TextVertexMain"}
		desc.fragment_func = {source = strings.clone_to_cstring(string(text_msl_frag_shader_code), context.temp_allocator), entry = "TextFragmentMain"}
		desc.uniform_blocks[0] = {stage = .VERTEX, size = size_of(SokolTextVertexUniforms), msl_buffer_n = 0}
		desc.uniform_blocks[1] = {stage = .FRAGMENT, size = size_of(SokolTextFragmentUniforms), msl_buffer_n = 0}
		desc.views[0] = {texture = {stage = .FRAGMENT, image_type = ._2D, sample_type = .FLOAT, msl_texture_n = 0}}
		desc.views[1] = {texture = {stage = .FRAGMENT, image_type = ._2D, sample_type = .UINT, msl_texture_n = 1}}
		desc.samplers[0] = {stage = .FRAGMENT, sampler_type = .NONFILTERING, msl_sampler_n = 0}
		desc.samplers[1] = {stage = .FRAGMENT, sampler_type = .NONFILTERING, msl_sampler_n = 1}
	case .GLES3:
		desc.vertex_func = {source = strings.clone_to_cstring(text_glsl300es_vert_shader_code, context.temp_allocator), entry = "main"}
		desc.fragment_func = {source = strings.clone_to_cstring(text_glsl300es_frag_shader_code, context.temp_allocator), entry = "main"}
		desc.attrs[0].glsl_name = "pos"
		desc.attrs[1].glsl_name = "tex"
		desc.attrs[2].glsl_name = "jac"
		desc.attrs[3].glsl_name = "bnd"
		desc.attrs[4].glsl_name = "col"
		desc.uniform_blocks[0] = {stage = .VERTEX, layout = .STD140, size = size_of(SokolTextVertexUniforms)}
		desc.uniform_blocks[0].glsl_uniforms[0] = {type = .MAT4, glsl_name = "mvp"}
		desc.uniform_blocks[0].glsl_uniforms[1] = {type = .FLOAT2, glsl_name = "viewport"}
		desc.uniform_blocks[1] = {stage = .FRAGMENT, layout = .STD140, size = size_of(SokolTextFragmentUniforms)}
		desc.uniform_blocks[1].glsl_uniforms[0] = {type = .FLOAT, glsl_name = "weightBoost"}
		desc.views[0] = {texture = {stage = .FRAGMENT, image_type = ._2D, sample_type = .FLOAT}}
		desc.views[1] = {texture = {stage = .FRAGMENT, image_type = ._2D, sample_type = .UINT}}
		desc.samplers[0] = {stage = .FRAGMENT, sampler_type = .NONFILTERING}
		desc.samplers[1] = {stage = .FRAGMENT, sampler_type = .NONFILTERING}
		desc.texture_sampler_pairs[0] = {stage = .FRAGMENT, view_slot = 0, sampler_slot = 0, glsl_name = "curveTexture"}
		desc.texture_sampler_pairs[1] = {stage = .FRAGMENT, view_slot = 1, sampler_slot = 1, glsl_name = "bandTexture"}
	case .D3D11:
		hlsl_source := strings.clone_to_cstring(string(text_hlsl_shader_code), context.temp_allocator)
		desc.vertex_func = {source = hlsl_source, entry = "TextVertexMain", d3d11_target = "vs_5_0"}
		desc.fragment_func = {source = hlsl_source, entry = "TextFragmentMain", d3d11_target = "ps_5_0"}
		desc.attrs[0] = {base_type = .FLOAT, hlsl_sem_name = "TEXCOORD", hlsl_sem_index = 0}
		desc.attrs[1] = {base_type = .FLOAT, hlsl_sem_name = "TEXCOORD", hlsl_sem_index = 1}
		desc.attrs[2] = {base_type = .FLOAT, hlsl_sem_name = "TEXCOORD", hlsl_sem_index = 2}
		desc.attrs[3] = {base_type = .FLOAT, hlsl_sem_name = "TEXCOORD", hlsl_sem_index = 3}
		desc.attrs[4] = {base_type = .FLOAT, hlsl_sem_name = "TEXCOORD", hlsl_sem_index = 4}
		desc.uniform_blocks[0] = {stage = .VERTEX, size = size_of(SokolTextVertexUniforms), hlsl_register_b_n = 0}
		desc.uniform_blocks[1] = {stage = .FRAGMENT, size = size_of(SokolTextFragmentUniforms), hlsl_register_b_n = 0}
		desc.views[0] = {texture = {stage = .FRAGMENT, image_type = ._2D, sample_type = .FLOAT, hlsl_register_t_n = 0}}
		desc.views[1] = {texture = {stage = .FRAGMENT, image_type = ._2D, sample_type = .UINT, hlsl_register_t_n = 1}}
	case:
		return {}
	}
	for i in 0..<5 do desc.attrs[i].base_type = .FLOAT
	desc.texture_sampler_pairs[0].stage = .FRAGMENT
	desc.texture_sampler_pairs[0].view_slot = 0
	desc.texture_sampler_pairs[0].sampler_slot = 0
	desc.texture_sampler_pairs[1].stage = .FRAGMENT
	desc.texture_sampler_pairs[1].view_slot = 1
	desc.texture_sampler_pairs[1].sampler_slot = 1
	return sg.make_shader(desc)
}

create_text_pipeline :: proc(shader: sg.Shader) -> sg.Pipeline {
	blend := sg.Blend_State{
		enabled = true,
		src_factor_rgb = .SRC_ALPHA,
		dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
		op_rgb = .ADD,
		src_factor_alpha = .ONE,
		dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
		op_alpha = .ADD,
	}
	return sg.make_pipeline({
		shader = shader,
		layout = {
			buffers = {0 = {stride = i32(size_of(engine.Text_Vertex))}},
			attrs = {
				0 = {format = .FLOAT4, offset = i32(offset_of(engine.Text_Vertex, pos))},
				1 = {format = .FLOAT4, offset = i32(offset_of(engine.Text_Vertex, tex))},
				2 = {format = .FLOAT4, offset = i32(offset_of(engine.Text_Vertex, jac))},
				3 = {format = .FLOAT4, offset = i32(offset_of(engine.Text_Vertex, bnd))},
				4 = {format = .FLOAT4, offset = i32(offset_of(engine.Text_Vertex, col))},
			},
		},
		index_type = .UINT32,
		primitive_type = .TRIANGLES,
		cull_mode = .NONE,
		colors = {0 = {blend = blend}},
	})
}

ui_ortho :: proc(width, height: f32) -> matrix[4, 4]f32 {
	m: matrix[4, 4]f32
	m[0, 0] = 2.0 / width
	m[1, 1] = -2.0 / height
	m[2, 2] = 1.0
	m[3, 3] = 1.0
	m[0, 3] = -1.0
	m[1, 3] = 1.0
	return m
}

text_ortho :: proc(width, height: f32) -> matrix[4, 4]f32 {
	m: matrix[4, 4]f32
	m[0, 0] = 2.0 / width
	m[1, 1] = -2.0 / height
	m[2, 2] = 1.0
	m[3, 0] = -1.0
	m[3, 1] = 1.0
	m[3, 3] = 1.0
	return m
}

matrix4_identity :: proc() -> matrix[4, 4]f32 {
	m: matrix[4, 4]f32
	m[0, 0] = 1
	m[1, 1] = 1
	m[2, 2] = 1
	m[3, 3] = 1
	return m
}

renderer_mesh :: proc(renderer: ^SokolRenderer, handle: engine.MeshHandle) -> ^SokolGpuMesh {
	index := int(handle)
	assert(0 <= index && index < len(renderer.meshes))
	return &renderer.meshes[index]
}

renderer_update_stats :: proc(renderer: ^SokolRenderer) {
	renderer.stats.mesh_count = len(renderer.meshes)
	renderer.stats.pipeline_count = int(renderer.pipeline.id != 0) + int(renderer.line_pipeline.id != 0) + int(renderer.sky_pipeline.id != 0) + int(renderer.ui_pipeline.id != 0) + int(renderer.text != nil && renderer.text.pipeline.id != 0)
}

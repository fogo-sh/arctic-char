package engine

// ===================================================
// Engine-owned subset of odin-slug's GPU Bezier text rendering code.
//
// An Odin implementation of Eric Lengyel's Slug algorithm for
// resolution-independent text and vector icon rendering. The fragment
// shader evaluates quadratic Bezier curves per-pixel using winding
// numbers for crisp, infinitely scalable glyphs at any size or rotation.
//
// This core package is graphics-API-agnostic. It handles:
//   - TTF font loading and glyph extraction (via stb_truetype)
//   - Bezier curve processing and band acceleration
//   - SVG path parsing for vector icons
//   - CPU-side vertex packing for glyph quads
//   - Text measurement and layout
//   - Text effects (rainbow, wobble, shake, rotation, etc.)
//
// Backends (in separate packages) handle GPU upload and rendering:
//   - slug/backends/vulkan  — Vulkan 1.x with GLSL 4.50 shaders
//   - slug/backends/opengl  — OpenGL 3.3 (compatible with Raylib/rlgl)
//   - slug/backends/raylib  — Raylib via rlgl
//
// Usage:
//   1. Create a Context
//   2. Load fonts with font_load / font_load_ascii
//   3. Process fonts with font_process (generates GPU texture data)
//   4. Pass Texture_Pack_Result to your backend for GPU upload
//   5. Per frame: begin() -> draw_text() / draw_icon() -> end()
//   6. Backend reads ctx.vertices[:vertex_count()] and draws
// ===================================================

import stbtt "vendor:stb/truetype"

// --- Constants ---

// Texture dimensions — must match kLogBandTextureWidth in the fragment shader.
// Both curve and band textures use this as their row width.
BAND_TEXTURE_WIDTH_LOG2 :: 12
BAND_TEXTURE_WIDTH :: 1 << BAND_TEXTURE_WIDTH_LOG2 // 4096

// Initial capacity for the per-font glyph map.
// Not a hard limit — the map grows as needed for any Unicode codepoint.
INITIAL_GLYPH_CAPACITY :: 256

// Maximum glyph quads per frame (one quad per visible glyph instance)
MAX_GLYPH_QUADS :: 4096
VERTICES_PER_QUAD :: 4
INDICES_PER_QUAD :: 6
MAX_GLYPH_VERTICES :: MAX_GLYPH_QUADS * VERTICES_PER_QUAD
MAX_GLYPH_INDICES :: MAX_GLYPH_QUADS * INDICES_PER_QUAD

// Maximum colored background rects per frame.
// Used by draw_rect and draw_text_highlighted. Backends draw these
// with a flat-color shader BEFORE the Slug text pass so they appear behind text.
MAX_RECTS :: 512

// Maximum simultaneously loaded fonts
MAX_FONT_SLOTS :: 4

// Pixels of quad expansion for antialiasing border
DILATION_SCALE :: f32(1.0)

// Maximum error for cubic-to-quadratic Bezier subdivision (in em-space units)
CUBIC_TO_QUAD_TOLERANCE :: f32(0.001)

// A bundle of text rendering properties for consistent styling across a UI.
// Define named styles as package-level constants and pass them to
// draw_text_styled / measure_text_styled to keep call sites clean:
//
//   BODY  :: Text_Style{size = 24, color = WHITE}
//   ALERT :: Text_Style{size = 24, color = RED, underline = true}
//
//   draw_text_styled(ctx, "Warning!", x, y, ALERT)
//
// font_slot:          Context font slot to draw with. 0 = primary font (zero-value default).
// size:               em-square height in pixels. Must be set; 0 renders nothing.
// color:              RGBA text color.
// underline:          Draw an underline at ~10% em below the baseline.
// strikethrough:      Draw a strikethrough at ~30% em above the baseline.
// underline_color:    Independent color for the underline. Zero-alpha (default) uses text color.
// strikethrough_color: Independent color for the strikethrough. Zero-alpha (default) uses text color.
//
// Both decorations can be true simultaneously — they are drawn in one pass.
Text_Style :: struct {
	font_slot:           int,
	size:                f32,
	color:               Color,
	underline:           bool,
	strikethrough:       bool,
	underline_color:     Color,
	strikethrough_color: Color,
}

// Per-glyph transform returned by a Glyph_Xform_Proc callback.
// The zero value is the identity transform — returning {} from a callback
// draws the glyph exactly as draw_text would, with no changes.
//
// Fields:
//   offset — pixel nudge from the natural pen position. {0,0} = no move.
//   scale  — multiplier on font_size. 0 is treated as 1.0 (identity).
//   angle  — rotation in radians around the glyph's visual center. 0 = upright.
//   color  — per-glyph color. alpha=0 inherits the color passed to draw_text_transformed.
Glyph_Xform :: struct {
	offset: [2]f32,
	scale:  f32,
	angle:  f32,
	color:  Color,
}

// Callback type for draw_text_transformed.
// Called once per glyph in the string.
//
//   char_idx — 0-based rune index in the string
//   ch       — the Unicode codepoint
//   pen_x    — natural x position where the glyph would be drawn (baseline-left)
//   y        — baseline y position
//   userdata — pointer passed through from the draw call (carry time, state, etc.)
//
// Return a Glyph_Xform to modify the glyph. Return {} for identity (no change).
Glyph_Xform_Proc :: #type proc(
	char_idx: int,
	ch: rune,
	pen_x, y: f32,
	userdata: rawptr,
) -> Glyph_Xform

// Common color constants.
WHITE :: Color{1.0, 1.0, 1.0, 1.0}
BLACK :: Color{0.0, 0.0, 0.0, 1.0}
RED :: Color{1.0, 0.0, 0.0, 1.0}
GREEN :: Color{0.0, 1.0, 0.0, 1.0}
BLUE :: Color{0.0, 0.0, 1.0, 1.0}
YELLOW :: Color{1.0, 1.0, 0.0, 1.0}
CYAN :: Color{0.0, 1.0, 1.0, 1.0}
MAGENTA :: Color{1.0, 0.0, 1.0, 1.0}
ORANGE :: Color{1.0, 0.6, 0.0, 1.0}
GRAY :: Color{0.5, 0.5, 0.5, 1.0}
LIGHT_GRAY :: Color{0.75, 0.75, 0.75, 1.0}
DARK_GRAY :: Color{0.25, 0.25, 0.25, 1.0}
TRANSPARENT :: Color{0.0, 0.0, 0.0, 0.0}

// A screen-space rectangle for GPU scissor clipping.
// Passed to backend flush procs to restrict rendering to a sub-region of the screen.
//
// Zero value (w <= 0 || h <= 0) means no scissor — full-screen rendering.
// All coordinates are in pixels, Y-down from the top-left corner of the window.
//
//   renderer-specific text flush can use Scissor_Rect{x=10, y=50, w=300, h=200}
Scissor_Rect :: struct {
	x, y, w, h: f32,
}

// --- Vertex Format ---
// Matches the 5x vec4 attribute layout in the vertex shader (locations 0-4).
// 80 bytes per vertex. All data for fragment-shader curve evaluation is packed
// here — the vertex shader just transforms and forwards it.

Vertex :: struct {
	pos: [4]f32, // .xy = screen position, .zw = dilation normal
	tex: [4]f32, // .xy = em-space texcoord, .zw = packed glyph/band texture coords
	jac: [4]f32, // 2x2 inverse Jacobian (screen-space -> em-space)
	bnd: [4]f32, // band transform: em coord -> band index via (coord * scale + offset)
	col: Color, // vertex color RGBA
}

// Simple position + color vertex for solid background rectangles.
// Used by draw_rect / draw_text_highlighted. Backends render these
// with a flat-color shader before the Slug text pass.
Rect_Vertex :: struct {
	pos: [2]f32,
	col: Color,
}

// --- Curve and Glyph Types ---

Bezier_Curve :: struct {
	p1: [2]f32, // Start point
	p2: [2]f32, // Control point (off-curve)
	p3: [2]f32, // End point
}

// Spatial acceleration band — a horizontal or vertical slice of a glyph.
// The fragment shader only evaluates curves overlapping the current pixel's band.
Band :: struct {
	curve_count: u16,
	data_offset: u16, // Offset into per-glyph curve index list
}

// Per-glyph data: metrics, curves, bands, and GPU texture coordinates.
Glyph_Data :: struct {
	// Metrics (em-space, normalized)
	bbox_min:      [2]f32,
	bbox_max:      [2]f32,
	advance_width: f32,
	left_bearing:  f32,

	// Bezier curves defining the glyph outline
	curves:        [dynamic]Bezier_Curve,

	// Band acceleration data
	h_bands:       [dynamic]Band, // Horizontal bands (Y-axis slices)
	v_bands:       [dynamic]Band, // Vertical bands (X-axis slices)
	h_curve_lists: [dynamic]u16, // Curve indices per horizontal band
	v_curve_lists: [dynamic]u16, // Curve indices per vertical band

	// GPU texture coordinates (set during packing)
	curve_tex_x:   u16,
	curve_tex_y:   u16,
	band_tex_x:    u16,
	band_tex_y:    u16,
	band_max_x:    u16, // Number of vertical bands - 1
	band_max_y:    u16, // Number of horizontal bands - 1

	// Band transform for shader
	band_scale:    [2]f32,
	band_offset:   [2]f32,

	// Identity
	codepoint:     rune,
	glyph_index:   i32,
	valid:         bool,
}

// Font loaded from a TTF file.
Font :: struct {
	info:       stbtt.fontinfo,
	font_data:  []u8, // Raw TTF bytes (must stay alive while font is in use)

	// Vertical metrics (em-space)
	ascent:     f32,
	descent:    f32,
	line_gap:   f32,
	em_scale:   f32, // 1.0 / units_per_em

	// Cap height in em-space (from OS/2 table sCapHeight, or 'H' bbox fallback).
	// Used by font_snap_size() to align cap-height to the pixel grid.
	// Zero means not available (no OS/2 v2+ table and no 'H' glyph).
	cap_height: f32,

	// Glyph cache keyed by codepoint (supports any Unicode codepoint)
	glyphs:     map[rune]Glyph_Data,
}

// --- Core Context ---
// GPU-agnostic rendering state. Backends embed or wrap this.

Context :: struct {
	// Loaded fonts
	fonts:           [MAX_FONT_SLOTS]Font,
	font_loaded:     [MAX_FONT_SLOTS]bool,
	font_count:      int,

	// Per-slot fallback font index. -1 means no fallback.
	// When get_glyph_fallback can't find a glyph in slot N, it looks
	// in font_fallback[N], then font_fallback[font_fallback[N]], etc.
	// Only followed in shared_atlas mode (see get_glyph_fallback).
	font_fallback:   [MAX_FONT_SLOTS]int,

	// Active font for drawing
	active_font_idx: int,

	// When true, all fonts share a single curve/band texture atlas.
	// Enables free font interleaving (no use_font switching restriction)
	// and single-draw-call rendering in backends.
	// Set automatically by fonts_process_shared().
	shared_atlas:    bool,

	// UI scale factor — multiplied into font sizes via scaled_size().
	// Set this to match your display DPI or user preference.
	// Default: 1.0 (set in begin()).
	ui_scale:        f32,

	// Camera offset in pixels. Applied to all glyph and rect draw calls,
	// shifting the entire canvas without touching layout coordinates.
	// Use set_camera() to pan; reset with set_camera(ctx, 0, 0).
	camera_x:        f32,
	camera_y:        f32,

	// Coverage weight boost — applies sqrt(coverage) in the fragment shader.
	// This is Eric Lengyel's SLUG_WEIGHT optical correction: it makes antialiased
	// edges slightly bolder/heavier, counteracting the "too thin" appearance that
	// linear-space or sRGB blending can produce. Perceptually sharper body text.
	// Default: false (disabled). Set to true for bolder text edges.
	weight_boost:    bool,

	// Per-frame vertex buffer (CPU side — backends upload to GPU)
	vertices:        [MAX_GLYPH_VERTICES]Vertex,
	quad_count:      u32,

	// Per-font quad ranges for batched draw calls (unused in shared_atlas mode)
	font_quad_start: [MAX_FONT_SLOTS]u32,
	font_quad_count: [MAX_FONT_SLOTS]u32,

	// Per-frame rect buffer for solid background rectangles.
	// Backends drain this with a flat-color shader BEFORE the Slug text pass.
	// Each rect is stored as 4 vertices (a quad); backends use the same
	// 0,1,2,2,3,0 index pattern as the glyph pass.
	rect_vertices:   [MAX_RECTS * VERTICES_PER_QUAD]Rect_Vertex,
	rect_count:      u32,
}

// --- Context operations ---

// Reset quad and rect counters for a new frame.
// Call before any draw_text/draw_icon/draw_rect calls.
begin :: proc(ctx: ^Context) {
	ctx.quad_count = 0
	ctx.rect_count = 0
	ctx.active_font_idx = 0
	ctx.font_quad_start = {}
	ctx.font_quad_count = {}
	ctx.font_quad_start[0] = 0
	if ctx.ui_scale == 0 do ctx.ui_scale = 1.0
}

// Finalize per-font quad ranges. Call after all draw calls, before backend flush.
end :: proc(ctx: ^Context) {
	prev := ctx.active_font_idx
	ctx.font_quad_count[prev] = ctx.quad_count - ctx.font_quad_start[prev]
}

// Switch to a different font slot. All subsequent draw calls use this font.
// In shared_atlas mode: free interleaving, no restrictions.
// In per-font mode: returns false if the slot already has quads
// (switching back would corrupt the batch layout).
use_font :: proc(ctx: ^Context, slot: int) -> bool {
	if slot < 0 || slot >= MAX_FONT_SLOTS do return false
	if !ctx.font_loaded[slot] do return false
	if slot == ctx.active_font_idx do return true

	// Shared atlas: just switch, no batch tracking needed
	if ctx.shared_atlas {
		ctx.active_font_idx = slot
		return true
	}

	// Per-font mode: reject switching back to a font that already has quads
	if ctx.font_quad_count[slot] > 0 do return false

	// Finalize previous font's quad range
	prev := ctx.active_font_idx
	ctx.font_quad_count[prev] = ctx.quad_count - ctx.font_quad_start[prev]

	// Start new font's range
	ctx.active_font_idx = slot
	ctx.font_quad_start[slot] = ctx.quad_count
	return true
}

// Register a loaded font into a context slot.
// slot must be in [0, MAX_FONT_SLOTS). Returns false if the slot is invalid.
register_font :: proc(ctx: ^Context, slot: int, font: Font) -> bool {
	if slot < 0 || slot >= MAX_FONT_SLOTS do return false
	ctx.fonts[slot] = font
	ctx.font_loaded[slot] = true
	ctx.font_fallback[slot] = -1
	if slot >= ctx.font_count {
		ctx.font_count = slot + 1
	}
	return true
}

// Set the fallback font for a given slot.
// When drawing with font_slot active, any glyph missing from that font
// will be looked up in fallback_slot instead. Chains are supported:
// if fallback_slot also lacks the glyph, its own fallback is tried in turn.
//
// Requires shared atlas mode. The shared atlas is enabled automatically by
// fonts_process_shared() — do not call font_process() per-slot when using
// fallback chains. In per-font atlas mode the fallback slot is registered
// but silently ignored during rendering, because glyphs from different font
// textures cannot be mixed in the same draw call.
//
// Call after register_font and before fonts_process_shared.
// Returns false if either slot is invalid or unloaded.
font_set_fallback :: proc(ctx: ^Context, font_slot, fallback_slot: int) -> bool {
	if font_slot < 0 || font_slot >= MAX_FONT_SLOTS do return false
	if fallback_slot < 0 || fallback_slot >= MAX_FONT_SLOTS do return false
	if !ctx.font_loaded[font_slot] do return false
	if !ctx.font_loaded[fallback_slot] do return false
	ctx.font_fallback[font_slot] = fallback_slot
	return true
}

// Look up a glyph, following the fallback chain when a codepoint is missing.
//
// In shared_atlas mode: walks font_fallback[] until the glyph is found or
// the chain ends. The returned glyph may belong to a different font slot than
// the active one, but because all fonts share the same atlas texture the
// texture coordinates are always valid.
//
// In per-font mode: only searches the active font, same as get_glyph().
// Fallback is skipped to avoid emitting quads with coordinates from a
// different font's texture while the active font's texture is bound.
@(private = "package")
get_glyph_fallback :: proc(ctx: ^Context, ch: rune) -> ^Glyph_Data {
	slot := ctx.active_font_idx
	for _ in 0 ..< MAX_FONT_SLOTS {
		if slot < 0 || slot >= MAX_FONT_SLOTS do break
		if !ctx.font_loaded[slot] do break
		g := get_glyph(&ctx.fonts[slot], ch)
		if g != nil do return g
		// In per-font mode don't cross texture boundaries.
		if !ctx.shared_atlas do break
		next := ctx.font_fallback[slot]
		if next < 0 do break
		slot = next
	}
	return nil
}

// Get pointer to the currently active font.
active_font :: proc(ctx: ^Context) -> ^Font {
	assert(ctx.active_font_idx >= 0 && ctx.active_font_idx < MAX_FONT_SLOTS)
	return &ctx.fonts[ctx.active_font_idx]
}

// Return the slot index of the currently active font (0–MAX_FONT_SLOTS-1).
// Use this when you need to save and restore the active slot around a
// use_font() call, without tracking it in your own state.
active_font_index :: proc(ctx: ^Context) -> int {
	return ctx.active_font_idx
}

// Set the UI scale factor. Affects all subsequent scaled_size() calls.
// Use this at startup to match display DPI, or at runtime for a user
// "text size" slider.  1.0 = no scaling, 2.0 = double size, etc.
set_ui_scale :: proc(ctx: ^Context, scale: f32) {
	ctx.ui_scale = scale if scale > 0 else 1.0
}

// Set the camera offset. All draw calls are translated by (x, y) pixels.
// Use for canvas pan effects. Call set_camera(ctx, 0, 0) to reset.
set_camera :: proc(ctx: ^Context, x, y: f32) {
	ctx.camera_x = x
	ctx.camera_y = y
}

// Apply the context's UI scale to a logical font size.
// Use this in draw and measure calls so text scales uniformly:
//   text_draw(ctx, "hello", x, y, scaled_size(ctx, 24), color)
//   w, h := measure_text(font, "hello", scaled_size(ctx, 24))
scaled_size :: proc(ctx: ^Context, font_size: f32) -> f32 {
	return font_size * ctx.ui_scale
}

// Number of vertices written this frame (for backend upload).
vertex_count :: proc(ctx: ^Context) -> u32 {
	return ctx.quad_count * VERTICES_PER_QUAD
}

// Look up a glyph by codepoint. Returns nil if not loaded.
get_glyph :: proc(font: ^Font, ch: rune) -> ^Glyph_Data {
	g, ok := &font.glyphs[ch]
	if !ok || !g.valid do return nil
	return g
}

// --- Cleanup ---

glyph_data_destroy :: proc(g: ^Glyph_Data) {
	delete(g.curves)
	delete(g.h_bands)
	delete(g.v_bands)
	delete(g.h_curve_lists)
	delete(g.v_curve_lists)
	g^ = {}
}

font_destroy :: proc(font: ^Font) {
	for _, &g in font.glyphs {
		glyph_data_destroy(&g)
	}
	delete(font.glyphs)
	delete(font.font_data)
	font^ = {}
}

// Unload a single font from a context slot, freeing its glyph data.
// The backend is responsible for releasing any associated GPU resources.
unload_font :: proc(ctx: ^Context, slot: int) {
	if slot < 0 || slot >= MAX_FONT_SLOTS do return
	if !ctx.font_loaded[slot] do return
	font_destroy(&ctx.fonts[slot])
	ctx.font_loaded[slot] = false
}

destroy :: proc(ctx: ^Context) {
	for i in 0 ..< MAX_FONT_SLOTS {
		if ctx.font_loaded[i] {
			font_destroy(&ctx.fonts[i])
			ctx.font_loaded[i] = false
		}
	}
}

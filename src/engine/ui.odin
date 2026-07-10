package engine

import "base:runtime"
import "core:fmt"
import "core:log"
import clay "../../vendor/clay-odin"

UI_COMMAND_CAPACITY :: 256

UiCommandKind :: enum {
	Rectangle,
	Border,
	Text,
	ScissorStart,
	ScissorEnd,
}

UiCommand :: struct {
	kind:          UiCommandKind,
	bounds:        [4]f32,
	color:         [4]f32,
	corner_radii:  [4]f32,
	border_widths: [4]f32,
	text:          string,
	font_size:     f32,
}

UiContext :: struct {
	allocator: runtime.Allocator,
	memory:    []u8,
	ctx:       ^clay.Context,
}

DebugHudData :: struct {
	enabled: bool,
	object_count: int,
	suzanne_count: int,
	object_capacity: int,
	player_position: Vec3,
	player_velocity: Vec3,
	player_grounded: bool,
	fixed_steps: int,
	physics_step_ms: f32,
	physics_collide_ms: f32,
	physics_solve_ms: f32,
	physics_pairs_ms: f32,
	physics_contacts: int,
	physics_awake_contacts: int,
	physics_tree_height: int,
}

ui_create :: proc(width, height: i32, allocator := context.allocator) -> UiContext {
	min_memory_size := clay.MinMemorySize()
	memory := make([]u8, int(min_memory_size), allocator)
	arena := clay.CreateArenaWithCapacityAndMemory(uint(min_memory_size), raw_data(memory))
	ctx := clay.Initialize(arena, {f32(width), f32(height)}, {handler = ui_error_handler})
	assert(ctx != nil)
	clay.SetMeasureTextFunction(ui_measure_text, nil)
	return {allocator = allocator, memory = memory, ctx = ctx}
}

ui_destroy :: proc(ui: ^UiContext) {
	if ui.memory != nil do delete(ui.memory, ui.allocator)
	ui^ = {}
}

ui_debug_hud_append_commands :: proc(ui: ^UiContext, win_size: [2]i32, debug: DebugHudData, timing: FrameTimingStats, out: ^[dynamic]UiCommand) {
	if !debug.enabled {
		return
	}

	clay.SetCurrentContext(ui.ctx)
	clay.SetLayoutDimensions({f32(win_size.x), f32(win_size.y)})
	clay.BeginLayout()

	if clay.UI(clay.ID("Root"))({
		layout = {
			sizing = {width = clay.SizingGrow({}), height = clay.SizingGrow({})},
			padding = {24, 24, 24, 24},
		},
	}) {
		if clay.UI(clay.ID("HudPanel"))({
			layout = {
				layoutDirection = .TopToBottom,
				sizing = {width = clay.SizingFixed(336), height = clay.SizingFit({})},
				padding = clay.PaddingAll(12),
				childGap = 7,
			},
			backgroundColor = {35, 31, 27, 210},
			cornerRadius = clay.CornerRadiusAll(12),
			border = {color = {206, 162, 98, 230}, width = clay.BorderOutside(2)},
		}) {
			clay.TextStatic("Debug", {
				textColor = {243, 226, 188, 255},
				fontSize = 22,
			})

			ui_debug_hud_row("Objects", 0, fmt.tprintf("%d / %d", debug.object_count, debug.object_capacity))
			ui_debug_hud_row("Suzannes", 1, fmt.tprintf("%d", debug.suzanne_count))
			ui_debug_hud_row("Player", 2, fmt.tprintf("%.2f %.2f %.2f", debug.player_position.x, debug.player_position.y, debug.player_position.z))
			ui_debug_hud_row("Velocity", 3, fmt.tprintf("%.2f %.2f %.2f", debug.player_velocity.x, debug.player_velocity.y, debug.player_velocity.z))
			ui_debug_hud_row("Grounded", 4, fmt.tprintf("%v", debug.player_grounded))
			ui_debug_hud_row("CPU frame", 5, fmt.tprintf("%.1f ms", timing.frame_ms))
			ui_debug_hud_row("Update", 6, fmt.tprintf("%.1f ms", timing.update_ms))
			ui_debug_hud_row("Render", 7, fmt.tprintf("%.1f ms", timing.render_ms))
			ui_debug_hud_row("Fixed steps", 8, fmt.tprintf("%d", debug.fixed_steps))
			ui_debug_hud_row("Box3D step", 9, fmt.tprintf("%.2f ms", debug.physics_step_ms))
			ui_debug_hud_row("Collide/solve", 10, fmt.tprintf("%.2f / %.2f", debug.physics_collide_ms, debug.physics_solve_ms))
			ui_debug_hud_row("Pairs", 11, fmt.tprintf("%.2f ms", debug.physics_pairs_ms))
			ui_debug_hud_row("Contacts", 12, fmt.tprintf("%d awake %d", debug.physics_contacts, debug.physics_awake_contacts))
			ui_debug_hud_row("Tree height", 13, fmt.tprintf("%d", debug.physics_tree_height))
		}
	}

	commands := clay.EndLayout(1.0 / 60.0)
	ui_append_clay_commands(&commands, out)
}

ui_debug_hud_row :: proc(label: string, index: u32, value: string) {
	if clay.UI(clay.ID("DebugRow", index))({
		layout = {
			layoutDirection = .LeftToRight,
			sizing = {width = clay.SizingGrow({}), height = clay.SizingFixed(26)},
			childGap = 10,
		},
	}) {
		if clay.UI(clay.ID("DebugLabel", index))({
			layout = {sizing = {width = clay.SizingFixed(96), height = clay.SizingFixed(24)}},
		}) {
			clay.Text(label, {textColor = {180, 170, 150, 255}, fontSize = 16})
		}
		if clay.UI(clay.ID("DebugValue", index))({
			layout = {sizing = {width = clay.SizingGrow({}), height = clay.SizingFixed(24)}},
		}) {
			clay.Text(value, {textColor = {243, 226, 188, 255}, fontSize = 16})
		}
	}
}

ui_error_handler :: proc "c" (error_data: clay.ErrorData) {
	context = default_context
	text := string(error_data.errorText.chars[:error_data.errorText.length])
	log.errorf("Clay error: %v: %s", error_data.errorType, text)
}

ui_measure_text :: proc "c" (text: clay.StringSlice, config: ^clay.TextElementConfig, user_data: rawptr) -> clay.Dimensions {
	_ = user_data
	font_size := f32(config.fontSize)
	return {width = f32(text.length) * font_size * 0.55, height = font_size}
}

ui_append_clay_commands :: proc(commands: ^clay.ClayArray(clay.RenderCommand), out: ^[dynamic]UiCommand) {
	for i in 0..<commands.length {
		command := clay.RenderCommandArray_Get(commands, i)
		#partial switch command.commandType {
		case .Rectangle:
			data := command.renderData.rectangle
			append(out, UiCommand {
				kind = .Rectangle,
				bounds = ui_bounds(command.boundingBox),
				color = ui_color(data.backgroundColor),
				corner_radii = ui_corner_radii(data.cornerRadius),
			})
		case .Border:
			data := command.renderData.border
			append(out, UiCommand {
				kind = .Border,
				bounds = ui_bounds(command.boundingBox),
				color = ui_color(data.color),
				corner_radii = ui_corner_radii(data.cornerRadius),
				border_widths = {f32(data.width.left), f32(data.width.right), f32(data.width.top), f32(data.width.bottom)},
			})
		case .ScissorStart:
			append(out, UiCommand{kind = .ScissorStart, bounds = ui_bounds(command.boundingBox)})
		case .ScissorEnd:
			append(out, UiCommand{kind = .ScissorEnd})
		case .Text:
			data := command.renderData.text
			append(out, UiCommand {
				kind = .Text,
				bounds = ui_bounds(command.boundingBox),
				color = ui_color(data.textColor),
				text = string(data.stringContents.chars[:data.stringContents.length]),
				font_size = f32(data.fontSize),
			})
		}
	}
}

ui_bounds :: proc(bounds: clay.BoundingBox) -> [4]f32 {
	return {bounds.x, bounds.y, bounds.width, bounds.height}
}

ui_color :: proc(color: clay.Color) -> [4]f32 {
	return {f32(color.r) / 255.0, f32(color.g) / 255.0, f32(color.b) / 255.0, f32(color.a) / 255.0}
}

ui_corner_radii :: proc(radii: clay.CornerRadius) -> [4]f32 {
	return {radii.topLeft, radii.topRight, radii.bottomLeft, radii.bottomRight}
}

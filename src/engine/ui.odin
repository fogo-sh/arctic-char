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
	prop_count: int,
	object_capacity: int,
	player_position: Vec3,
	player_velocity: Vec3,
	player_grounded: bool,
	command_sequence: u32,
	acked_command: u32,
	prediction_error: f32,
	prediction_replay_count: u32,
	prediction_correction_count: u32,
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
				sizing = {width = clay.SizingFixed(246), height = clay.SizingFit({})},
				padding = clay.PaddingAll(8),
				childGap = 4,
			},
			backgroundColor = {35, 31, 27, 170},
			cornerRadius = clay.CornerRadiusAll(8),
			border = {color = {206, 162, 98, 180}, width = clay.BorderOutside(1)},
		}) {
			clay.TextStatic("Net Debug", {
				textColor = {243, 226, 188, 255},
				fontSize = 16,
			})

			ui_debug_hud_row("Objs", 0, fmt.tprintf("%d/%d", debug.object_count, debug.object_capacity))
			ui_debug_hud_row("Prop", 1, fmt.tprintf("%d", debug.prop_count))
			ui_debug_hud_row("Pos", 2, fmt.tprintf("%.1f %.1f %.1f", debug.player_position.x, debug.player_position.y, debug.player_position.z))
			ui_debug_hud_row("Vel", 3, fmt.tprintf("%.1f %.1f %.1f", debug.player_velocity.x, debug.player_velocity.y, debug.player_velocity.z))
			ui_debug_hud_row("Cmd", 4, fmt.tprintf("%d ack %d", debug.command_sequence, debug.acked_command))
			ui_debug_hud_row("Pred", 5, fmt.tprintf("err %.3f replay %d", debug.prediction_error, debug.prediction_replay_count))
			ui_debug_hud_row("Corr", 6, fmt.tprintf("%d", debug.prediction_correction_count))
			ui_debug_hud_row("Frame", 7, fmt.tprintf("%.1fms", timing.frame_ms))
			ui_debug_hud_row("Phys", 8, fmt.tprintf("%d %.1fms", debug.fixed_steps, debug.physics_step_ms))
		}

		ui_debug_hud_render_samples()
	}

	commands := clay.EndLayout(1.0 / 60.0)
	ui_append_clay_commands(&commands, out)
}

ui_debug_hud_render_samples :: proc() {
	if clay.UI(clay.ID("RenderSamples"))({
		layout = {
			layoutDirection = .TopToBottom,
			sizing = {width = clay.SizingFixed(246), height = clay.SizingFit({})},
			padding = clay.PaddingAll(8),
			childGap = 6,
		},
		backgroundColor = {18, 22, 28, 160},
		cornerRadius = clay.CornerRadiusAll(10),
		border = {color = {120, 170, 220, 180}, width = clay.BorderOutside(1)},
	}) {
		clay.TextStatic("Render Samples", {textColor = {219, 234, 254, 255}, fontSize = 14})
		if clay.UI(clay.ID("SampleRoundedRow"))({
			layout = {
				layoutDirection = .LeftToRight,
				sizing = {width = clay.SizingGrow({}), height = clay.SizingFixed(34)},
				childGap = 6,
			},
		}) {
			ui_debug_hud_sample_box("SampleR2", 2, {210, 132, 70, 220})
			ui_debug_hud_sample_box("SampleR6", 6, {83, 160, 220, 220})
			ui_debug_hud_sample_box("SampleR14", 14, {95, 190, 130, 220})
		}

		if clay.UI(clay.ID("SampleBorderRow"))({
			layout = {
				layoutDirection = .LeftToRight,
				sizing = {width = clay.SizingGrow({}), height = clay.SizingFixed(34)},
				childGap = 6,
			},
		}) {
			ui_debug_hud_border_box("SampleB1", 1)
			ui_debug_hud_border_box("SampleB2", 2)
			ui_debug_hud_border_box("SampleB4", 4)
		}

		if clay.UI(clay.ID("SampleClip"))({
			layout = {
				sizing = {width = clay.SizingGrow({}), height = clay.SizingFixed(24)},
				padding = {5, 5, 4, 4},
			},
			clip = {horizontal = true, vertical = true},
			backgroundColor = {7, 12, 18, 210},
			cornerRadius = clay.CornerRadiusAll(5),
			border = {color = {240, 220, 160, 220}, width = clay.BorderOutside(1)},
		}) {
			clay.TextStatic("Clipped text should stop at the panel edge >>>>>>>>>>>", {textColor = {255, 238, 180, 255}, fontSize = 12})
		}
	}
}

ui_debug_hud_sample_box :: proc(id: string, radius: f32, color: clay.Color) {
	if clay.UI(clay.ID(id))({
		layout = {sizing = {width = clay.SizingFixed(40), height = clay.SizingFixed(28)}},
		backgroundColor = color,
		cornerRadius = clay.CornerRadiusAll(radius),
	}) {}
}

ui_debug_hud_border_box :: proc(id: string, width: u16) {
	if clay.UI(clay.ID(id))({
		layout = {sizing = {width = clay.SizingFixed(40), height = clay.SizingFixed(28)}},
		backgroundColor = {24, 30, 38, 230},
		cornerRadius = clay.CornerRadiusAll(8),
		border = {color = {245, 245, 245, 230}, width = clay.BorderOutside(width)},
	}) {}
}

ui_debug_hud_row :: proc(label: string, index: u32, value: string) {
	if clay.UI(clay.ID("DebugRow", index))({
		layout = {
			layoutDirection = .LeftToRight,
			sizing = {width = clay.SizingGrow({}), height = clay.SizingFixed(18)},
			childGap = 6,
		},
	}) {
		if clay.UI(clay.ID("DebugLabel", index))({
			layout = {sizing = {width = clay.SizingFixed(48), height = clay.SizingFixed(18)}},
		}) {
			clay.Text(label, {textColor = {180, 170, 150, 255}, fontSize = 12})
		}
		if clay.UI(clay.ID("DebugValue", index))({
			layout = {sizing = {width = clay.SizingGrow({}), height = clay.SizingFixed(18)}},
		}) {
			clay.Text(value, {textColor = {243, 226, 188, 255}, fontSize = 12})
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

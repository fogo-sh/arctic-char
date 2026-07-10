package engine

import "base:runtime"
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

ui_placeholder_append_commands :: proc(ui: ^UiContext, win_size: [2]i32, out: ^[dynamic]UiCommand) {
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
				sizing = {width = clay.SizingFixed(260), height = clay.SizingFixed(148)},
				padding = clay.PaddingAll(12),
				childGap = 10,
			},
			backgroundColor = {35, 31, 27, 210},
			cornerRadius = clay.CornerRadiusAll(12),
			border = {color = {206, 162, 98, 230}, width = clay.BorderOutside(2)},
		}) {
			clay.TextStatic("Arctic Char", {
				textColor = {243, 226, 188, 255},
				fontSize = 24,
			})

			ui_placeholder_bar("BarA", 0, 196, 167, 91)
			ui_placeholder_bar("BarB", 1, 102, 128, 121)

			if clay.UI(clay.ID("ClippedStrip"))({
				layout = {
					sizing = {width = clay.SizingGrow({}), height = clay.SizingFixed(42)},
				},
				backgroundColor = {28, 28, 26, 220},
				cornerRadius = clay.CornerRadiusAll(8),
				clip = {horizontal = true, vertical = true},
			}) {
				if clay.UI(clay.ID("OverflowBlob"))({
					layout = {sizing = {width = clay.SizingFixed(340), height = clay.SizingFixed(42)}},
					backgroundColor = {157, 48, 59, 175},
					cornerRadius = clay.CornerRadiusAll(8),
				}) {}
			}
		}
	}

	commands := clay.EndLayout(1.0 / 60.0)
	ui_append_clay_commands(&commands, out)
}

ui_placeholder_bar :: proc(id: string, index: u32, r, g, b: f32) {
	if clay.UI(clay.ID(id, index))({
		layout = {
			sizing = {width = clay.SizingGrow({}), height = clay.SizingFixed(26)},
		},
		backgroundColor = {r, g, b, 210},
		cornerRadius = clay.CornerRadiusAll(7),
		border = {color = {255, 252, 240, 80}, width = clay.BorderOutside(1)},
	}) {}
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

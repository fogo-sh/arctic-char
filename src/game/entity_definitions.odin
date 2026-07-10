package game

EntityEditorKind :: enum {
	Point,
	Solid,
}

EntityRuntimeKind :: enum {
	Ignore,
	PropSuzanne,
	SpawnerSuzanne,
	TriggerTeleport,
}

EntityPropertyType :: enum {
	String,
	Integer,
	Float,
	TargetSource,
	TargetDestination,
}

EntityPropertyDef :: struct {
	name:          string,
	label:         string,
	type:          EntityPropertyType,
	default_value: string,
	description:   string,
}

EntityDef :: struct {
	classname:    string,
	description:  string,
	editor_kind:  EntityEditorKind,
	runtime_kind: EntityRuntimeKind,
	color:        Vec3,
	size_min:     Vec3,
	size_max:     Vec3,
	properties:   []EntityPropertyDef,
}

ENTITY_WORLDSPAWN_PROPERTIES := [?]EntityPropertyDef {
	{name = "message", label = "Map Title", type = .String, description = "Optional display name for the map."},
}

ENTITY_ANGLE_PROPERTIES := [?]EntityPropertyDef {
	{name = "angle", label = "Yaw", type = .Integer, default_value = "0", description = "Yaw in editor degrees."},
}

ENTITY_TARGETNAME_PROPERTIES := [?]EntityPropertyDef {
	{name = "targetname", label = "Target Name", type = .TargetSource, description = "Name other entities can target."},
}

ENTITY_TARGET_PROPERTIES := [?]EntityPropertyDef {
	{name = "target", label = "Target", type = .TargetDestination, description = "Target entity name."},
}

ENTITY_SPAWNER_SUZANNE_PROPERTIES := [?]EntityPropertyDef {
	{name = "interval", label = "Spawn Interval", type = .Float, default_value = "0.35", description = "Seconds between Suzanne spawns."},
	{name = "count", label = "Spawn Count", type = .Integer, default_value = "4", description = "Maximum number of Suzannes to spawn."},
}

ENTITY_TRIGGER_TELEPORT_PROPERTIES := [?]EntityPropertyDef {
	{name = "target", label = "Target", type = .TargetDestination, description = "Teleport destination targetname."},
	{name = "target_origin", label = "Direct Target Origin", type = .String, description = "Optional direct teleport destination as x y z editor coordinates."},
	{name = "target_angle", label = "Direct Target Yaw", type = .Integer, default_value = "0", description = "Yaw used with target_origin."},
}

ENTITY_DEFINITIONS := [?]EntityDef {
	{
		classname = "worldspawn",
		description = "World entity",
		editor_kind = .Solid,
		runtime_kind = .Ignore,
		color = {180, 180, 180},
		properties = ENTITY_WORLDSPAWN_PROPERTIES[:],
	},
	{
		classname = "player",
		description = "Player start",
		editor_kind = .Point,
		runtime_kind = .Ignore,
		color = {0, 255, 0},
		size_min = {-16, -16, -24},
		size_max = {16, 16, 32},
		properties = ENTITY_ANGLE_PROPERTIES[:],
	},
	{
		classname = "direction_light",
		description = "Directional light placeholder",
		editor_kind = .Point,
		runtime_kind = .Ignore,
		color = {255, 220, 120},
		size_min = {-8, -8, -8},
		size_max = {8, 8, 8},
		properties = ENTITY_ANGLE_PROPERTIES[:],
	},
	{
		classname = "info_teleport_destination",
		description = "Teleport destination",
		editor_kind = .Point,
		runtime_kind = .Ignore,
		color = {80, 200, 255},
		size_min = {-8, -8, -8},
		size_max = {8, 8, 8},
		properties = ENTITY_TARGETNAME_PROPERTIES[:],
	},
	{
		classname = "prop_suzanne",
		description = "Dynamic Suzanne prop",
		editor_kind = .Point,
		runtime_kind = .PropSuzanne,
		color = {255, 150, 80},
		size_min = {-16, -16, -16},
		size_max = {16, 16, 16},
		properties = nil,
	},
	{
		classname = "spawner_suzanne",
		description = "Timed Suzanne spawner",
		editor_kind = .Point,
		runtime_kind = .SpawnerSuzanne,
		color = {255, 90, 180},
		size_min = {-12, -12, -12},
		size_max = {12, 12, 12},
		properties = ENTITY_SPAWNER_SUZANNE_PROPERTIES[:],
	},
	{
		classname = "trigger_teleport",
		description = "Teleports the player when touched",
		editor_kind = .Solid,
		runtime_kind = .TriggerTeleport,
		color = {80, 160, 255},
		properties = ENTITY_TRIGGER_TELEPORT_PROPERTIES[:],
	},
}

entity_def_find :: proc(classname: string) -> (^EntityDef, bool) {
	for &def in ENTITY_DEFINITIONS {
		if def.classname == classname {
			return &def, true
		}
	}
	return nil, false
}

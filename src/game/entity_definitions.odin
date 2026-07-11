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

ENTITY_PROP_SUZANNE_PROPERTIES := [?]EntityPropertyDef {
	{name = "net_policy", label = "Network Policy", type = .String, default_value = "server", description = "server for authoritative synced prop, client for presentation-only local prop."},
}

ENTITY_TRIGGER_TELEPORT_PROPERTIES := [?]EntityPropertyDef {
	{name = "target", label = "Target", type = .TargetDestination, description = "Teleport destination targetname."},
	{name = "target_origin", label = "Direct Target Origin", type = .String, description = "Optional direct teleport destination as x y z editor coordinates."},
	{name = "target_angle", label = "Direct Target Yaw", type = .Integer, default_value = "0", description = "Yaw used with target_origin."},
}

ENTITY_FUNC_TRAIN_PROPERTIES := [?]EntityPropertyDef {
	{name = "target", label = "First Path Corner", type = .TargetDestination, description = "Initial path_corner target. The train starts with its bounds center at this corner."},
	{name = "targetname", label = "Target Name", type = .TargetSource, description = "Optional name for trigger-controlled trains."},
	{name = "speed", label = "Speed", type = .Float, default_value = "100", description = "Movement speed in Quake units per second."},
	{name = "wait", label = "Wait", type = .Float, default_value = "0", description = "Seconds to wait at path corners unless overridden by the corner."},
}

ENTITY_PATH_CORNER_PROPERTIES := [?]EntityPropertyDef {
	{name = "targetname", label = "Target Name", type = .TargetSource, description = "Name used by func_train or the previous path_corner."},
	{name = "target", label = "Next Path Corner", type = .TargetDestination, description = "Next path_corner in the route. Leave empty to stop."},
	{name = "speed", label = "Speed Override", type = .Float, default_value = "0", description = "Optional train speed after this corner; 0 keeps the current speed."},
	{name = "wait", label = "Wait Override", type = .Float, default_value = "0", description = "Optional seconds to wait at this corner."},
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
		properties = ENTITY_PROP_SUZANNE_PROPERTIES[:],
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
	{
		classname = "func_train",
		description = "Moving brush platform following path_corner entities",
		editor_kind = .Solid,
		runtime_kind = .Ignore,
		color = {180, 120, 255},
		properties = ENTITY_FUNC_TRAIN_PROPERTIES[:],
	},
	{
		classname = "path_corner",
		description = "Point target for func_train movement; origin marks the train bounds center",
		editor_kind = .Point,
		runtime_kind = .Ignore,
		color = {220, 180, 255},
		size_min = {-8, -8, -8},
		size_max = {8, 8, 8},
		properties = ENTITY_PATH_CORNER_PROPERTIES[:],
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

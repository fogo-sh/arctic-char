#+test
package game

import "core:testing"

@(test)
test_entity_def_find_known_entity :: proc(t: ^testing.T) {
	def, ok := entity_def_find("spawner_suzanne")
	testing.expect(t, ok, "spawner_suzanne should be registered")
	testing.expect_value(t, def.runtime_kind, EntityRuntimeKind.SpawnerSuzanne)
	testing.expect_value(t, def.editor_kind, EntityEditorKind.Point)
}

@(test)
test_entity_def_find_unknown_entity :: proc(t: ^testing.T) {
	def, ok := entity_def_find("totally_not_real")
	testing.expect(t, !ok, "unknown entity should not resolve")
	testing.expect(t, def == nil, "unknown entity should return nil def")
}

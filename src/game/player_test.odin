#+test
package game

import "core:testing"

@(test)
test_player_teleport_preserves_respawn_state :: proc(t: ^testing.T) {
	spawn_position := Vec3{1, 2, 3}
	spawn_yaw := f32(0.25)
	player := player_create(spawn_position, spawn_yaw)
	player.velocity = {4, 5, 6}
	player.pitch = 0.5
	player.grounded = true
	player.ground_normal = {0, 0, 1}

	teleport_position := Vec3{-7, 8, -9}
	teleport_yaw := f32(1.5)
	player_teleport(&player, teleport_position, teleport_yaw)

	testing.expect_value(t, player.position, teleport_position)
	testing.expect_value(t, player.yaw, teleport_yaw)
	testing.expect_value(t, player.velocity, Vec3{})
	testing.expect_value(t, player.pitch, f32(0))
	testing.expect_value(t, player.grounded, false)
	testing.expect_value(t, player.ground_normal, Vec3{0, 1, 0})
	testing.expect_value(t, player.spawn_position, spawn_position)
	testing.expect_value(t, player.spawn_yaw, spawn_yaw)
}

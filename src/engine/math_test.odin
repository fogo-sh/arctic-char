#+test
package engine

import "core:math"
import "core:testing"

TEST_EPSILON :: f32(0.0001)

@(test)
test_matrix4_perspective_z0_maps_near_and_far_to_sdl_depth_range :: proc(t: ^testing.T) {
	m := matrix4_perspective_z0_f32(math.to_radians(f32(90)), 1, 0.1, 100)

	near_clip := m * [4]f32{0, 0, -0.1, 1}
	far_clip := m * [4]f32{0, 0, -100, 1}

	testing.expect(t, test_f32_near(near_clip.z / near_clip.w, 0), "near plane should map to NDC z=0")
	testing.expect(t, test_f32_near(far_clip.z / far_clip.w, 1), "far plane should map to NDC z=1")
}

test_f32_near :: proc(a, b: f32) -> bool {
	return math.abs(a - b) <= TEST_EPSILON
}

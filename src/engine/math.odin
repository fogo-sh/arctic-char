package engine

import "core:math"

// SDL GPU expects clip-space/NDC depth in [0, 1]. Odin's core linalg
// perspective helpers follow the OpenGL-style [-1, 1] depth convention and
// only expose a z-axis flip, not a GLM-style depth-range switch. Keep the
// renderer projection explicit so the shader outputs SDL GPU-compatible clip
// coordinates without hidden API fixups.
matrix4_perspective_z0_f32 :: proc(fovy, aspect, near, far: f32) -> matrix[4, 4]f32 {
	tan_half_fovy := math.tan(0.5 * fovy)
	m: matrix[4, 4]f32
	m[0, 0] = 1 / (aspect * tan_half_fovy)
	m[1, 1] = 1 / tan_half_fovy
	m[3, 2] = 1
	m[2, 2] = far / (far - near)
	m[2, 3] = -(far * near) / (far - near)
	m[2] = -m[2]
	return m
}

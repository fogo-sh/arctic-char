# Engine Decisions

This file records engine-level choices that should stay consistent across code,
assets, tools, and future agent work. Keep entries short and update them when a
choice changes.

## Coordinate And Projection Conventions

- Engine world space uses meters internally and +Y as up.
- Player/camera forward is currently -Z at yaw 0.
- Imported assets, authored maps, and generated collision should be converted at
  load/import boundaries into engine world space.
- Renderer projection must output SDL GPU-compatible clip coordinates.
- SDL GPU clip/NDC depth is [0, 1], so engine projection helpers should say this
  explicitly in their names, e.g. `matrix4_perspective_z0_f32`.
- Do not use Odin's generic `linalg.matrix4_perspective_f32` directly for the SDL
  GPU render path unless its clip-space convention is deliberately adapted.
- Keep shader-side coordinate fixups rare. Prefer named engine math helpers or
  import-time conversion so transforms remain inspectable.

Current findings:

- Odin `core:math/linalg` perspective helpers expose `flip_z_axis`, but not a
  GLM-style `DEPTH_ZERO_TO_ONE` switch.
- Odin's generic perspective helper uses the OpenGL-style depth formula that maps
  to NDC Z [-1, 1].
- SDL GPU's rasterizer state has `enable_depth_clip`; set it explicitly for the
  main pipeline so far/near plane intersections clip instead of depth-clamping.

## Renderer Boundary

- Scene code decides what should be drawn and emits `RenderItem` values.
- Renderer code owns GPU resources, render targets, pipeline state, and command
  recording.
- Renderer code should stay generic: it knows about meshes, render items, and
  matrices, not semantic game objects.

This project is written in Odin, using SDL3 GPU sandbox.

Keep changes explicit, inspectable, and useful for learning
how the rendering, physics, and asset pieces fit together.

## Architecture

The code is split by ownership and learning area:

- `main.odin`: tiny executable entrypoint that imports `src/game/`
- `src/engine/app.odin`: SDL lifecycle, event loop, timing, input polling, and draw orchestration
- `src/engine/fs.odin`: qpath search paths, file reads, and modification-time polling
- `src/engine/mesh.odin`: CPU mesh loading and shared vertex data types
- `src/engine/graphics.odin`: platform shader selection and shader creation
- `src/engine/renderer.odin`: SDL GPU pipeline, render targets, mesh upload/replacement, and draw pass
- `src/engine/physics.odin`: generic Box3D world ownership, fixed stepping, mesh cooking, transform conversion
- `src/game/game.odin`: package-level game startup and engine callback wiring
- `src/game/assets.odin`: scene asset loading and collision asset policy
- `src/game/scene.odin`: scene objects, spawning, map reload application, and render item extraction
- `src/game/scene_physics.odin`: game collision layers, map collision body, and prop body creation
- `src/game/collision_shape.odin`: Suzanne convex hull cooking
- `src/game/player.odin`: Quake-style player controller
- `src/game/player_mover.odin`: Box3D kinematic player mover
- `src/game/map.odin`, `src/game/map_mesh.odin`, `src/game/level.odin`: Valve 220 map parsing and level compilation
- `tools/make_collision_mesh.py`: Blender collision mesh generator

Prefer small, named functions that keep ownership clear. Do not hide important
engine steps behind generic abstractions too early.

## Object Model

Use a simple fat-struct object model:

```odin
Object :: struct {
	id:        ObjectId,
	name:      string,
	kind:      ObjectKind,
	transform: Transform,
	render:    RenderObject,
	physics:   PhysicsObject,
}
```

## Renderer Boundaries

The renderer should stay generic. It should know about `MeshHandle`, `GpuMesh`,
and `RenderItem`, not semantic game object names.

Scene code owns the decision to draw an object. Renderer code owns GPU resource
creation and command recording.

## Memory Management

Follow Odin-style lifetime management:

- Long-lived owned data should record its allocator where useful, e.g. `CpuMesh`,
  `Scene`, and `Renderer`.
- Short-lived scratch allocations can use `context.temp_allocator`, but scope
  them with `runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()` unless they are tied to
  an explicit frame-level reset.
- Do not per-frame allocate dynamic arrays just because the temp allocator exists.
  Prefer reusable storage when the maximum size is known.
- `App.render_items` is reusable frame storage preallocated by the engine.
- `Scene.objects` is preallocated to the known maximum object count.
- Avoid putting growing dynamic arrays into custom arenas casually; arena-backed
  growth can leave old backing buffers behind. Use arenas when the group of
  allocations truly shares a lifetime.

## Physics And Collision

Physics uses Odin's vendored Box3D binding.

Keep physics world ownership in `PhysicsWorld`. `Object.physics.body` is a handle
to a Box3D world-owned body, not an independently owned allocation.

## Verification

For code changes, run:

```sh
just build
```

For rendering/physics-affecting changes, also run a short smoke test:

```sh
./build/arctic-char & pid=$!; sleep 5; kill $pid; wait $pid || true
```

For shader changes, run:

```sh
just check-shaders
```

This project is written in Odin, using SDL3 GPU sandbox.

Keep changes explicit, inspectable, and useful for learning
how the rendering, physics, and asset pieces fit together.

## Architecture

The code is split by ownership and learning area:

- `main.odin`: tiny executable entrypoint that imports `src/game/`
- `src/engine/app.odin`: SDL lifecycle, event loop, timing, input polling, and draw orchestration
- `src/engine/fs.odin`: qpath search paths, file reads, and modification-time polling
- `src/engine/math.odin`: renderer-facing math helpers with explicit graphics API conventions
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
- `src/hot_reload/main.odin`: development host that loads `build/hot_reload/game` as a platform dynamic library
- `tools/make_collision_mesh.py`: Blender collision mesh generator

Prefer small, named functions that keep ownership clear. Do not hide important
engine steps behind generic abstractions too early.

Engine-level choices and conventions live in `docs/engine-decisions.md`. Read and
update it when changing coordinate systems, renderer math, asset import policy,
or other cross-cutting decisions.

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

## Hot Reload

The normal executable still imports `src/game/`. The development hot-reload host
imports only `src/engine/`, loads `build/hot_reload/game` as `.dll`, `.dylib`, or
`.so`, and resolves the exported `game_*` procs.

Current dynamic-library reload policy preserves `Game_State` but rebuilds Box3D state. The
old library receives `game_before_hot_reload`, syncs body transforms/velocities into
plain scene data, and destroys the old Box3D world/assets. The host keeps old
libraries loaded, swaps the function table, then calls `game_hot_reloaded` so the
new dylib can recreate the Box3D world, map body, and prop bodies from scene data.
Do not preserve raw Box3D handles across dylib reload while game code still calls
Box3D directly.

## Verification

For code changes, run:

```sh
python cli.py test
python cli.py build
```

Run build-style commands sequentially, not in parallel. `smoke`, `net-smoke`,
`mp-test` without `--no-build`, `hot-build`, and `build` may touch shared files
under `build/`; concurrent invocations can race on generated intermediates.

Clay is vendored as a C header and built into a static library under `build/`.
Do not rebuild it manually unless `vendor/clay/clay.h` changed or `build/` was
cleaned; `cli.py` skips the Clay compile when the archive is already current.

For rendering/physics-affecting changes, also run a short smoke test:

```sh
python cli.py smoke
```

For shader changes, run:

```sh
python cli.py check-shaders
```

For hot-reload host changes, run:

```sh
python cli.py hot-build
```

## Debugging The Running App

Prefer measurements from the real executable over guesses. The app logs through
`core:log` and SDL's log callback, so CLI runs and smoke tests capture useful
startup, renderer, asset, map, and profiling messages.

Use the normal executable for direct runs:

```sh
python cli.py build
python cli.py run -- --map test
```

Use longer smoke runs to reproduce time-based problems such as spawning cliffs,
physics spirals, or hot-path regressions:

```sh
python cli.py smoke --seconds 20
```

When adding temporary performance diagnostics, keep them narrow and named. Useful
runtime counters include object count, draw count, triangle count, fixed-step
count, per-system wall time, and Box3D's own `World_GetProfile` and
`World_GetCounters` values. For physics cliffs, log object/contact counts next to
`Profile.step`, `Profile.collide`, `Profile.solve`, and the number of fixed steps
processed in the frame. A spike where fixed steps climb above one usually means a
spiral-of-death: one fixed step exceeded the frame budget, so the accumulator is
trying to catch up by doing more physics work per frame.

On macOS, use Xcode Instruments from the CLI with `xcrun xctrace`. First inspect
available templates if needed:

```sh
xcrun xctrace list templates
```

Capture a Time Profiler trace by launching the built executable directly:

```sh
xcrun xctrace record \
  --template "Time Profiler" \
  --time-limit 20s \
  --no-prompt \
  --output "build/arctic-char-time-profile.trace" \
  --target-stdout - \
  --launch -- "build/arctic-char" --map test
```

Export the table of contents and then the time-profile rows for headless grepping
or offline analysis:

```sh
xcrun xctrace export \
  --input "build/arctic-char-time-profile.trace" \
  --toc \
  --output "build/arctic-char-time-profile-toc.xml"

xcrun xctrace export \
  --input "build/arctic-char-time-profile.trace" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  --output "build/arctic-char-time-profile.xml"
```

Open the `.trace` in Instruments when call-tree navigation is useful:

```sh
open build/arctic-char-time-profile.trace
```

For GPU investigations, the same workflow works with templates such as
`Game Performance`, `Game Performance Overview`, or `Metal System Trace`. Keep GPU
and CPU conclusions separate: a low GPU time with high frame time usually means
the app is CPU-bound or blocking on synchronization, not shader/raster work.

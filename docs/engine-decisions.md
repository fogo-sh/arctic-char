# Engine Decisions

This file records engine-level choices that should stay consistent across code,
assets, tools, and future agent work. Keep entries short and update them when a
choice changes.

## Coordinate And Projection Conventions

- Engine world space uses meters internally and +Y as up.
- Player/camera forward is currently -Z at yaw 0.
- Quake units are converted into meters at load/compile boundaries. The current
  scale is `QU_TO_M = 0.038`.
- Imported assets, authored maps, and generated collision should be converted at
  load/import boundaries into engine world space.
- Do not make gameplay or physics code think in graphics API clip-space terms.
  Keep API-specific conventions at the renderer/projection boundary.
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

## Map And Asset Loading

- Editable source assets stay authoritative during development.
- Valve 220 `.map` files are the current level-authoring source format.
- Map loading is split into parsing and compilation:
  - `map.odin` parses entities, brushes, faces, properties, and texture axes.
  - `map_mesh.odin` turns brush faces into renderable mesh data.
  - `level.odin` owns the compiled level asset used by game startup/reload.
- Runtime code should consume compiled level data instead of reparsing source
  text in unrelated systems.
- TrenchBroom support is generated from canonical Odin entity metadata, not from
  ad-hoc FGD files. Update `src/game/entity_definitions.odin` first, then
  regenerate the profile.
- Collision assets may use separate authored meshes when that keeps physics
  simpler and more stable than deriving collision from visual meshes.
- Future compiled/binary asset formats should be disposable build products. The
  source text/assets remain the inspectable truth unless we explicitly change
  that policy.

## Renderer Boundary

- Scene code decides what should be drawn and emits `RenderItem` values.
- Renderer code owns GPU resources, render targets, pipeline state, and command
  recording.
- Renderer code should stay generic: it knows about meshes, render items, and
  matrices, not semantic game objects.
- `RenderItem` currently carries a `MeshHandle` and model matrix. Keep semantic
  object names, entity kinds, and gameplay decisions outside the renderer.
- The renderer owns SDL GPU depth target creation, MSAA target creation,
  immutable mesh upload/replacement, and draw pass recording.
- The current main pass uses a `D32_FLOAT` depth target, clears depth to `1.0`,
  and uses `.LESS` depth testing/writes.
- Shader-side coordinate fixups should be avoided unless they are clearly local
  to a shader feature. Prefer explicit engine math helpers so projection choices
  remain visible in Odin code.
- Clay UI is vendored as source in `vendor/clay` and `vendor/clay-odin`; generated
  Clay static libraries are local build products under `build/clay` created by
  `python cli.py clay-lib`.
- The first Clay renderer path is deliberately minimal: SDL GPU rectangles,
  borders, scissors, and text. Images and custom commands are deferred until
  there is a concrete UI need.
- Clay render commands should be translated into engine-owned `UiCommand` values
  before they reach the generic renderer. Keep Clay types out of `renderer.odin`.
- UI drawing currently happens at the end of the main render pass, using the same
  color/depth target setup as the world pass but with depth testing disabled.
- UI text uses engine-owned Slug-style TTF/glyph packing code in `src/engine`;
  renderer-owned SDL GPU code uploads the curve/band textures and draws text from
  engine-owned `UiCommand.Text` values.
- Text shader source stays in HLSL under `shaders/hlsl/text.hlsl` and is compiled
  through `shadercross` like the other shader domains.
- The placeholder UI is currently part of the normal render path. Replace it with
  game-owned UI commands when real HUD/menu needs become concrete.

## Sky And Fog

- The current art direction favors simple, non-textured graphics.
- The sky is procedural: a fullscreen gradient pass using Flexoki-inspired blues
  instead of a cubemap or sky texture.
- Distant world geometry fades into the sky/fog color in the world fragment
  shader. This hides hard far-plane clipping and gives distance cues.
- Keep the fog color matched to the sky horizon/clear color. If these diverge,
  the far-plane fade illusion becomes obvious.
- Prefer simple distance fog first. Cubemaps, image skyboxes, and atmospheric
  scattering are intentionally deferred until there is a concrete need.

## Physics Ownership

- Physics uses Odin's vendored Box3D binding.
- `PhysicsWorld` owns the Box3D world. Scene objects store `b3.BodyId` handles;
  they do not own Box3D body allocations.
- Game collision layers and object-specific body creation live in game code, not
  the generic engine physics wrapper.
- Fixed stepping is owned by the scene/game update path. Do not advance Box3D
  opportunistically from rendering code.
- The fixed gameplay/physics step is currently 40 Hz. Dedicated server simulation
  should advance from its own fixed tick, not once per received input packet.
- Physics transforms crossing into renderer data should use explicit conversion
  helpers, e.g. `physics_body_matrix`.

## Networking Ownership

- Multiplayer should use a server-authoritative client/server model first, not
  deterministic lockstep or full physics rollback.
- Use Odin's official `vendor:ENet` binding through a small shared `src/net`
  wrapper rather than exposing ENet structs across game code.
- Keep the game protocol above the transport boundary so Steam Networking
  Sockets can become a later backend after the ENet handshake/input/snapshot path
  proves the required API shape.
- Keep transport concerns in `src/net`, renderer/window concerns in `src/engine`,
  and protocol/session/snapshot decisions in game code.
- The server owns authoritative scene simulation and Box3D state. Clients send
  inputs and render snapshots, adding local-player prediction only after the
  basic authoritative snapshot path works.
- Dedicated servers assign player ids, queue deduplicated user commands per
  accepted session, drain queued commands in sequence order on the 40 Hz server
  tick, and broadcast full player snapshots independently of input arrival.
- Local play should not have an offline simulation fork. With no `--connect`, the
  app uses an in-process loopback session: client input is serialized as
  `User_Cmd`, copied through bounded in-memory packet queues, parsed by the
  shared protocol, and applied by the local authoritative scene before rendering.
- The initial protocol smoke validates protocol version, map name, and a simple
  content id before accepting a client into later gameplay state.
- See `docs/multiplayer.md` for the current milestone plan.

## Hot Reload

- The normal executable imports `src/game` directly.
- The development host imports only `src/engine` and loads the game dynamic
  library from `build/hot_reload/game` with the platform extension.
- Hot reload preserves `Game_State` memory when the game memory size still
  matches and the game does not request a forced restart.
- Raw Box3D handles are not preserved across dynamic-library reloads while game
  code still calls Box3D directly.
- Before reload, the old game library syncs runtime physics state back into plain
  scene data, then destroys the old Box3D world/assets.
- The host keeps old libraries loaded, swaps the resolved function table, and
  calls the new library's hot-reload hook to rebuild Box3D world state, map body,
  collision assets, and prop bodies from plain scene data.
- If hot reload policy changes, update both the host/game code and this document.

## Tooling And Verification

- `cli.py` is the canonical local task runner. Do not reintroduce parallel task
  definitions in a `justfile` or shell scripts unless there is a specific need.
- `mise.toml` pins the expected Odin and Python tool versions.
- `python cli.py build`, `build-release`, and hot-reload game builds regenerate
  the current platform's Clay static library before invoking Odin.
- For code changes, run `python cli.py build`.
- For rendering or physics-affecting changes, also run `python cli.py smoke`.
- For shader changes, run `python cli.py check-shaders`.
- Shader source files are split by domain under `shaders/hlsl`. Avoid adding
  unrelated entrypoints to a shared shader source if it causes generated-artifact
  churn in other pipelines.
- For hot-reload host changes, run `python cli.py hot-build`.
- For entity metadata or TrenchBroom profile changes, regenerate with
  `python cli.py trenchbroom-profile`.

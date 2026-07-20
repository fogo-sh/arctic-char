# System Map

This is the current high-level map of the engine. It is intentionally more useful
than exhaustive: follow these links first when trying to understand a system.

## Entrypoints

- [`main.odin`](../main.odin): normal native SDL client executable.
- [`src/server/main.odin`](../src/server/main.odin): dedicated server process shell.
- [`src/hot_reload/main.odin`](../src/hot_reload/main.odin): development hot-reload host.

## Engine Runtime

- [`src/engine`](../src/engine): platform-independent engine contracts and helpers.
- [`src/engine_sdl`](../src/engine_sdl): SDL app loop and SDL GPU renderer.
- [`src/engine/app_api.odin`](../src/engine/app_api.odin): callback API between app runtime and game code.
- [`src/engine/render_api.odin`](../src/engine/render_api.odin): renderer API exposed to game asset upload code.
- [`src/engine/render_types.odin`](../src/engine/render_types.odin): render items, debug lines, UI commands, and pass globals.
- [`src/engine_sdl/app.odin`](../src/engine_sdl/app.odin): SDL lifecycle, input polling, frame timing, and fixed update orchestration.
- [`src/engine_sdl/renderer.odin`](../src/engine_sdl/renderer.odin): SDL GPU resource ownership and world render pass.

## Game Simulation

- [`src/game/game.odin`](../src/game/game.odin): game callback implementation and top-level `Game_State`.
- [`src/game/scene.odin`](../src/game/scene.odin): object storage, player records, scene stepping, render item extraction, and hot-reload scene rebuilds.
- [`src/game/scene_entities.odin`](../src/game/scene_entities.odin): map entity spawning, trigger behavior, and simple think/touch logic.
- [`src/game/scene_physics.odin`](../src/game/scene_physics.odin): game collision layers, map body creation, prop bodies, and trigger bodies.
- [`src/game/assets.odin`](../src/game/assets.odin): level and prop asset loading policy.

## Movement And Physics

- [`src/game/player.odin`](../src/game/player.odin): Quake-style player movement inputs and controller state.
- [`src/game/player_mover.odin`](../src/game/player_mover.odin): Box3D-backed kinematic capsule movement.
- [`src/engine/physics.odin`](../src/engine/physics.odin): generic Box3D world helpers and fixed-step ownership.
- [`src/game/collision_shape.odin`](../src/game/collision_shape.odin): prop hull cooking policy.

## Maps And Entities

- [`src/game/map.odin`](../src/game/map.odin): Valve 220 `.map` parsing.
- [`src/game/map_mesh.odin`](../src/game/map_mesh.odin): brush-to-render-mesh compilation.
- [`src/game/level.odin`](../src/game/level.odin): loaded level asset wrapper and diagnostics.
- [`src/game/entity_definitions.odin`](../src/game/entity_definitions.odin): editor/runtime entity metadata.

## Multiplayer

- [`src/protocol`](../src/protocol): packet headers, user commands, snapshots, and packet codecs.
- [`src/net`](../src/net): ENet transport wrapper with no game semantics.
- [`src/game/net_server.odin`](../src/game/net_server.odin): authoritative server/session core used by dedicated server and local loopback.
- [`src/game/net_client.odin`](../src/game/net_client.odin): client connection state, user command generation, prediction, reconciliation, and snapshot application.
- [`src/game/replication.odin`](../src/game/replication.odin): replicated prop transform buffers and collision proxy helpers.
- [`src/server/net_impairment.odin`](../src/server/net_impairment.odin): local downstream network impairment for testing snapshot behavior.

## Documentation

- [`src/*/doc.odin`](../src): package-level documentation consumed by `odin doc`.
- [`docs/runtime-flow.md`](runtime-flow.md): curated flow explanation.
- [`docs/multiplayer.md`](multiplayer.md): multiplayer architecture notes and prior art.
- [`docs/engine-decisions.md`](engine-decisions.md): conventions that should be updated when changing architecture.

# Runtime Flow

This project is a native SDL engine with a Source-like, multiplayer-first shape.
Local play still uses client/server semantics so the single-player-feeling path
does not become a separate game architecture.

## Entrypoints

- `main.odin` is the normal SDL executable. It imports `src/game` and starts the
  engine app loop.
- `src/server/main.odin` is the headless dedicated server. It owns ENet process
  setup and drives the same `game.NetServer` core used by local loopback.
- `src/hot_reload/main.odin` is a development host. It reloads game code but does
  not define a second runtime model.

## SDL App Loop

`src/engine/app.odin` owns wall-clock time, SDL lifecycle, window events, input
polling, and renderer orchestration. It calls game callbacks rather than owning
game rules.

Per frame, the app loop does this:

1. Poll SDL events and update input state.
2. Accumulate elapsed wall time.
3. Run fixed updates while enough accumulated time remains.
4. Ask game code for render items and UI commands.
5. Submit one SDL GPU render pass.

The app may render at any frame rate. Gameplay advances in fixed steps.

## Game Loop

`src/game/game.odin` wires engine callbacks to package-level game systems. It is
the handoff point between generic engine code and game-owned state.

The game owns:

- current map and scene data
- local or connected network mode
- player movement and prediction
- authoritative server state for loopback
- render item extraction from scene objects

## Scene And Physics

`src/game/scene.odin` owns objects. An object is a fat struct with identity,
transform, render state, physics state, and optional replication state.

`src/engine/physics.odin` owns generic Box3D world helpers. Game code decides
which objects get bodies and which collision layers they use.

Physics runs at a fixed rate. Server simulation is authoritative for multiplayer
state. Client-side prop bodies for replicated props are collision proxies used by
prediction, not independent gameplay authority.

## Networking Flow

The server is authoritative. Clients send input; the server sends snapshots.

Client to server:

1. Client samples local input into `protocol.User_Cmd`.
2. Recent unacknowledged commands are packed for redundancy.
3. ENet or loopback transport delivers those commands to `game.NetServer`.
4. The server consumes at most one command per accepted player per server tick.

Server to client:

1. `game.NetServer` steps the authoritative scene.
2. Snapshot code gathers player and prop states.
3. Snapshots are byte-budgeted into clusters.
4. Clients apply authoritative state, reconcile the local player, and interpolate
   remote entities for rendering.

## Tick Rates

- Server simulation: `NET_SERVER_TICK_HZ`, currently 64 Hz.
- Physics stepping: `PHYSICS_STEP_HZ`, currently 128 Hz.
- Snapshot send rate: `NET_SNAPSHOT_HZ`, currently 32 Hz.
- Remote interpolation delay: time-based, currently 100 ms.

Keep these rates explicit. Do not hide them behind a generic timer abstraction;
the point is to make authority, prediction, and rendering easy to reason about.

## Engine Scope

The engine should support a real multiplayer game before it grows more backends.
Near-term engine functionality should stay focused on:

- native SDL window/input/GPU rendering
- dedicated server and local loopback using the same authoritative server core
- map loading, static collision, and entity spawning
- player movement, prediction, reconciliation, and remote interpolation
- server-owned dynamic props with clear client proxy rules
- debug HUD/logging for ticks, packet flow, prediction error, and physics cost
- simple game UI for menus, HUD, and connection state
- asset reload/restart workflow that does not obscure ownership
- audio only when there is concrete gameplay demand

Deferred unless there is a direct game need:

- browser builds
- alternate renderer/application backends
- networking transports beyond ENet
- full dynamic rigid-body rollback
- speculative ECS or plugin abstractions

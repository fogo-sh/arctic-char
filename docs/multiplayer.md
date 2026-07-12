# Multiplayer Architecture Notes

This is the current working plan for adding multiplayer without mixing transport
details into game simulation code too early. ENet is the first transport because
it is small, direct-IP friendly, and already available through Odin's vendor
packages.

## Direction

- Use a server-authoritative client/server model.
- Clients send compact player input commands to the server.
- The server owns the authoritative scene, fixed-step simulation, Box3D world,
  and spawn/despawn decisions.
- Clients render server snapshots with interpolation for remote objects.
- Add local-player prediction and reconciliation after the first snapshot path is
  working.
- Local play uses client/server semantics. When `--connect` is omitted, the app
  starts an in-process server core and connects the normal client path to it.
  Local input is serialized as `User_Cmd`, parsed by the same server core used by
  the dedicated executable, simulated on the same fixed tick, and returned to the
  client as normal server packets. Loopback uses separate client and server
  scenes; only the in-process transport is special. External `--connect` uses
  ENet as the transport feeding that same core.
- Do not try deterministic lockstep or full Box3D rollback first. Box3D dynamic
  prop state, contacts, and collision islands make that a much larger project.

## Package Boundaries

- `src/net` should wrap Odin's official `vendor:ENet` package and expose
  transport-level concepts: host creation, connect/listen, poll events, send
  packet, disconnect, peer handle, channel id, and packet reliability.
- The Odin package name for this directory is `transport`, not `net`, because
  `net` conflicts with Odin's `core:net` package when both are imported.
- Keep `src/net` separate from `src/engine` so the dedicated server can use the
  transport without importing SDL, GPU, renderer, UI, or client input code.
- Transport networking should not know about maps, players, object kinds,
  snapshots, prediction, or game entity ids.
- `src/game/net*.odin` should own game protocol details: hello/version/map checks,
  stable network ids, player input commands, server snapshots, spawn/despawn
  messages, interpolation buffers, and reconciliation.
- `src/game/net_server.odin` is the canonical authoritative server/session core.
  The dedicated server executable should be a transport and process wrapper, not
  a second server implementation.
- Shared wire-format tests live in `src/protocol`. Runtime smoke tests should use
  the real app client and dedicated server so temporary network bridge
  executables do not become a second client implementation.
- The scene remains the owner of gameplay and physics. Networking code should
  drive scene inputs and consume scene state, not replace scene ownership.

## ENet Choice

- Prefer Odin's official `vendor:ENet` binding over vendoring a third-party Odin
  binding.
- Keep a project wrapper anyway so the rest of the codebase does not depend on
  ENet's C-shaped API, packet lifetime rules, or raw peer structs.
- Keep the first wrapper synchronous and frame-polled. Do not add a networking
  thread until the main-thread version has real pressure.

## Steam Networking Later

- Steam Networking Sockets is a good later transport backend, but not the first
  implementation target.
- Valve's `ISteamNetworkingSockets` API is message-oriented, supports reliable
  and unreliable messages, handles splitting/coalescing, includes encryption and
  authentication, and can route P2P traffic through Steam Datagram Relay.
- Steam Datagram Relay helps with NAT traversal, IP hiding, and sometimes route
  quality. It also introduces Steam account/app setup and binding maturity risk.
- Keep the game protocol above the transport so Steam Networking can replace or
  sit beside ENet without changing snapshots, user commands, interpolation, or
  prediction code.
- Revisit Steam once the ENet path has a working handshake, input stream, and
  snapshot stream. At that point the transport abstraction will have real usage
  pressure instead of speculative API shape.

## Channels

- Channel 0, reliable: connection handshake, protocol version, map name/hash,
  initial world state, spawn/despawn, disconnect reasons.
- Channel 1, unreliable sequenced: client input commands.
- Channel 2, unreliable sequenced: server snapshots.
- Optional channel 3, unreliable or reliable by message kind: ping, stats, and
  debug instrumentation.

## Tick Model

- Keep gameplay simulation fixed-step, but keep timing policies separate.
- Authoritative server simulation currently runs at 64 Hz.
- Box3D physics substeps currently run at up to 128 Hz inside each scene step.
- Server snapshots currently send at 32 Hz, not every server tick.
- Remote interpolation delay is specified in time, currently 100 ms, then
  converted to server ticks for the current tick rate.
- Add a monotonically increasing server tick.
- Client input messages should include client input sequence and the client tick
  or sampled time they were generated from.
- One `User_Cmd` sequence currently represents one fixed server tick. Clients
  generate commands from a `NET_SERVER_TICK_TIME` accumulator and packetize recent
  unacknowledged commands for redundancy.
- The server consumes at most one queued command per accepted player per server
  tick. `last_processed_user_cmd` identifies the exact command represented by the
  resulting authoritative player state.
- Server snapshots should include the authoritative server tick and the last input
  sequence processed for each relevant player.
- Clients should keep a small interpolation buffer for remote entities and render
  behind the latest server snapshot by a short delay.

## Prior Art To Study

The following repositories were cloned for inspection under
`/var/folders/pn/yr8zyjld29v3rsxwmpx_pfrh0000gn/T/opencode/multiplayer-prior-art`.
Treat them as references for shape and tradeoffs, not code to copy wholesale.

## id Tech Networking References

These are the best external writeups found so far. Prefer reading these before
diving directly into the id source trees.

### Fabien Sanglard, Quake 3 Network Model

- URL: `https://fabiensanglard.net/quake3/network.php`
- Focus: Quake 3 snapshots, delta compression, NetChannel, and the snapshot
  history diagrams.
- Key lessons:
  - Fast-action games should not retransmit stale world snapshots. Send a newer
    snapshot instead.
  - The server keeps a per-client ring of recent snapshots.
  - The client acknowledges the newest snapshot it successfully received.
  - The next server snapshot is delta-compressed against the last acknowledged
    snapshot, not necessarily the immediately previous snapshot.
  - If the acknowledged baseline is missing or too old, the server falls back to a
    zero/dummy baseline and effectively sends a full snapshot.
  - This single algorithm both compresses state and naturally recovers from packet
    loss by including old unacknowledged changes alongside new changes.
  - Avoid IP/router fragmentation. Keep packets around MTU-sized chunks instead
    of assuming large UDP datagrams are safe.
- Application here:
  - Add per-client `SnapshotHistory` once we move past transport smoke tests.
  - Include `snapshot_sequence` and `ack_snapshot_sequence` in our protocol even
    before we implement bit-level delta compression.
  - Treat ENet reliable delivery as a tool for small control messages, not the
    core snapshot transport.

### Jacek Fedorynski, Quake 3 Network Protocol

- URL: `https://www.jfedor.org/quake3/`
- Focus: Practical protocol-level explanation of Quake 3 packets, user commands,
  reliable commands, snapshot deltas, Huffman compression, and debugging tools.
- Key lessons:
  - Normal client packets contain user commands plus reliable client commands.
  - Normal server packets contain snapshots plus reliable server commands.
  - Reliable command streams are sequence-numbered and acknowledged by the other
    side, then resent until acknowledged.
  - User commands are not resent forever. Instead, recent commands are duplicated
    a small fixed number of times in later client packets.
  - Snapshots are unreliable but sequence-numbered, and snapshot acks matter for
    delta compression.
  - Client-visible entity sets are filtered; the server does not send every entity
    to every client.
- Application here:
  - Use explicit packet headers with protocol version, connection/session id,
    packet sequence, ack fields, message kind, and payload length.
  - Start with simple byte structs, but keep the packet shape compatible with
    later bit-packing and delta compression.
  - Add debug counters like packet loss, last snapshot, acked snapshot, duplicated
    input count, and extrapolated/interpolated frame counts.

### Quake 3 Unlagged Networking Primer

- URL: `https://www.ra.is/unlagged/network.html`
- Focus: Clear terminology and runtime flow for commands, snapshots,
  interpolation, extrapolation, prediction, and prediction error.
- Key lessons:
  - A command is a compact input sample: timestamp, view angles, buttons, weapon,
    and intended movement.
  - The server accepts player commands as they arrive, while non-player objects
    advance on server frames.
  - The client renders from snapshots, interpolating between two known snapshots
    when possible and extrapolating only when it must.
  - The local player is special: predict local movement ahead from the latest
    authoritative player state plus unacknowledged commands.
- Application here:
  - Split replicated state into `PlayerState` for the local authoritative player
    and `EntityState` for other renderable objects.
  - Keep local-player prediction separate from remote-entity interpolation.
  - Record when the client is interpolating vs extrapolating in the HUD.

### Gaffer On Games, What Every Programmer Needs To Know About Game Networking

- URL: `https://gafferongames.com/post/what_every_programmer_needs_to_know_about_game_networking/`
- Focus: Broad explanation of lockstep, client/server, client-side prediction,
  and server reconciliation.
- Key lessons:
  - Avoid deterministic peer-to-peer lockstep for this project. Complex physics and
    floating-point behavior make exact determinism fragile.
  - Server-authoritative client/server gives cheat resistance and permits late
    join, at the cost of snapshot bandwidth and prediction complexity.
  - Prediction correction should rewind to the authoritative state, discard older
    inputs, then replay unacknowledged inputs.
- Application here:
  - Preserve one shared movement function for server and client prediction paths.
  - Do not let clients submit positions as authoritative state.

### Gaffer On Games, Snapshot Compression

- URL: `https://gafferongames.com/post/snapshot_compression/`
- Focus: Modern snapshot compression for many dynamic physics objects.
- Key lessons:
  - High-rate full snapshots become prohibitively expensive with many dynamic
    objects.
  - Delta compression requires the sender to know which baseline snapshot the
    receiver has.
  - Changed-object lists can beat one changed bit per object when most objects are
    unchanged.
  - Quantized positions, velocities, and compressed orientations are cumulative
    wins after the protocol shape is stable.
- Application here:
  - Start player-only, then add a capped dynamic prop set.
  - Do not attempt to stream all stress-test Suzanne props until packet budgets are
    visible.
  - Delay aggressive bit-packing until the semantic packet model is proven.

### Quake 2 Documentation Project, Networking Data Flow

- URL: `https://www.gamers.org/dEngine/quake2/Q2DP/Q2DP_Network/Q2DP_Network.shtml`
- Focus: Quake/QuakeWorld/Quake 2 client-server architecture and prediction data
  flow.
- Key lessons:
  - QuakeWorld's important step was not generic dead reckoning. It was POV latency
    compensation through local movement prediction and clipping.
  - Quake 2 still separates the authoritative world from the client-side predicted
    world image.
  - Multiple world images exist at once: authoritative server state, simplified
    network state, client prediction state, and render state.
- Application here:
  - Name these layers explicitly in Odin instead of letting scene/render/network
    state blur together.
  - Keep render interpolation state separate from authoritative scene state.

### Yahn Bernier, Latency Compensating Methods

- URL: `https://www.gamedevs.org/uploads/latency-compensation-in-client-server-protocols.pdf`
- Focus: Valve/Half-Life client prediction and lag compensation.
- Key lessons:
  - Store each user command and the time it was generated.
  - Use the last server-acknowledged movement as the prediction base.
  - Share identical movement code between client and server where possible.
  - Hitscan/weapon lag compensation is a separate problem from movement prediction.
- Application here:
  - Initial milestone should only predict movement. Weapon/hitscan rewind should be
    a later design note, not part of ENet bring-up.

### J.M.P. van Waveren, The DOOM III Network Architecture

- URL: `https://fabiensanglard.net/doom3_documentation/The-DOOM-III-Network-Architecture.pdf`
- Focus: Doom 3's id Tech 4 network architecture, improving on Quake 3.
- Key lessons:
  - Snapshots and player input remain unreliable/high-frequency traffic.
  - Reliable messages are small and piggyback on the unreliable stream until
    acknowledged.
  - Doom 3 moves beyond Quake 3's fixed entity-state layout by allowing entities to
    write their own state into bit messages.
  - Doom 3 maintains a synchronized common base for entity state between server and
    each client, improving deltas when entities leave and re-enter PVS.
  - The server tells clients about duplicated user commands and timing so clients
    can tune prediction lead.
- Application here:
  - Quake 3-style last-acked snapshot deltas are simpler and should come first.
  - Doom 3-style common-base state is worth considering later if entities often
    enter/leave interest sets and deltas become poor.
  - Keep reliable protocol messages small and ordered ahead of snapshot processing.

### Original Source Trees

- Quake 2: `https://github.com/id-Software/Quake-2`
- Quake 3: `https://github.com/id-Software/Quake-III-Arena`
- Doom 3: `https://github.com/id-Software/DOOM-3`
- Local clones: `.../idtech-netcode/Quake-2`,
  `.../idtech-netcode/Quake-III-Arena`, `.../idtech-netcode/DOOM-3`
- Most useful Quake 3 files after reading the articles:
  - `code/server/sv_snapshot.c`
  - `code/qcommon/msg.c`
  - `code/client/cl_input.c`
  - `code/client/cl_parse.c`
  - `code/cgame/cg_snapshot.c`
  - `code/cgame/cg_predict.c`
- Most useful Quake 2 files:
  - `qcommon/net_chan.c`
  - `qcommon/pmove.c`
  - `client/cl_input.c`
  - `client/cl_pred.c`
  - `client/cl_parse.c`
- Most useful Doom 3 files:
  - `neo/framework/async/NetworkSystem.cpp`
  - `neo/game/Game_network.cpp`
  - `neo/d3xp/Game_network.cpp`

## id Tech Lessons For This Project

- Dedicated server and client should be separate entrypoints into shared gameplay
  and protocol code. This mirrors Quake 3's explicit server logical side and
  client presentation side.
- ENet should be treated as the UDP transport and connection helper, not as the
  whole game protocol. We still need our own sequence numbers, snapshot acks,
  message kinds, and per-client protocol state.
- Do not send gameplay snapshots over reliable delivery. Packet loss should be
  handled by the next snapshot deltaing against the last acknowledged baseline.
- Keep reliable traffic small: handshake, map changes, spawn/despawn if needed,
  chat/debug/control messages. Resend or use reliable ENet channels only for these
  low-rate messages.
- Add `UserCmd` as a first-class type. It should contain command sequence/time,
  movement axes, look delta or view angles, buttons, and selected action/weapon.
- Client packets should include a small run of recent user commands, not just the
  latest command.
- Server snapshots should include a snapshot sequence, server tick/time, last
  processed user command for the client, and enough state to reconstruct the
  client's current view.
- Server should keep per-client snapshot history. A fixed 32-slot ring is a good
  starting point because it mirrors Quake 3 and is easy to mask/index.
- Snapshot encoding should eventually compare current state against the last
  client-acknowledged baseline. The first implementation can send full structs
  while preserving the sequence/ack/baseline shape.
- Prediction should only cover the local player at first. Other players and props
  should render from interpolated snapshots.
- id Tech 2/3 prediction does not simulate arbitrary remote physics props locally.
  Client movement traces against solid entities from the active snapshot image.
  Quake 3 also adjusts the predicted player by the movement of the current ground
  mover after replaying commands.
- Source SDK 2013 has the same broad split: prediction replays local commands,
  interpolation remains presentation state, and before prediction it moves the
  local player's ground entity back to its last received network position so
  client movement does not collide against render-interpolated transforms.
- For this project, replicated dynamic props should therefore provide a
  non-simulating collision image for local prediction, but should not run full
  client-side rigid-body simulation. Full rollback of dynamic prop physics remains
  out of scope.
- Physics props are expensive and chaotic. Replicate a curated/capped set first;
  do not replicate every dynamic stress-test object by default.
- Add network diagnostics early: packet sequence, snapshot sequence, acked
  snapshot, input queue length, duplicate input count, packet loss, RTT, snapshot
  bytes, interpolation buffer depth, extrapolated frames, and prediction error.

### geckosio/snapshot-interpolation

- Repository: `https://github.com/geckosio/snapshot-interpolation`
- Local clone: `.../multiplayer-prior-art/snapshot-interpolation`
- Useful files:
  - `src/snapshot-interpolation.ts`
  - `src/vault.ts`
  - `src/lerp.ts`
  - `src/slerp.ts`
  - `example/server/index.js`
  - `example/client/index.js`
- Extract for this project:
  - Keep snapshot interpolation as a small data structure, not a framework.
  - Store snapshots in a bounded vault/ring buffer.
  - Render from `now - interpolation_buffer`, where the buffer is roughly a few
    server ticks.
  - Find the two snapshots bracketing render time and compute interpolation `t`.
  - Interpolate only fields that are declared interpolatable, e.g. position and
    orientation.
- Do not extract:
  - JSON cloning, dynamic field lookup, string-based parameter selection, or JS
    object shape flexibility. Odin should use fixed packet/state structs.

### grazianobolla/godot-monke-net

- Repository: `https://github.com/grazianobolla/godot-monke-net`
- Local clone: `.../multiplayer-prior-art/godot-monke-net`
- Useful files:
  - `addons/monke-net/src/Shared/Network/NetworkManagerEnet.cs`
  - `addons/monke-net/src/Shared/NetworkMessages/NetworkMessages.cs`
  - `addons/monke-net/src/ClientSide/InternalComponents/ClientInputManager.cs`
  - `addons/monke-net/src/ServerSide/InternalComponents/ServerInputReceiver.cs`
  - `addons/monke-net/src/ClientSide/InternalComponents/ClientSnapshotInterpolator.cs`
  - `addons/monke-net/src/ClientSide/InternalComponents/ClientPredictionManager.cs`
  - `addons/monke-net/src/ClientSide/Nodes/ClientPredictedEntity.cs`
  - `addons/monke-net/src/ClientSide/Nodes/ClientInterpolatedEntity.cs`
  - `addons/monke-net/src/ServerSide/Nodes/ServerStateSyncronizer.cs`
- Extract for this project:
  - Redundant input sending: include the current input plus unacknowledged recent
    inputs in each unreliable input packet.
  - Server input queue keyed by tick and controlled entity.
  - Server fallback to last input when an expected input is missing.
  - Client drops acknowledged inputs when snapshots arrive.
  - Prediction history stores tick, input, and enough local entity state to compare
    against authoritative snapshots.
  - Reconciliation flow: find matching predicted tick, compare authoritative vs
    predicted state, restore authoritative state, then replay remaining inputs.
  - Debug counters are useful from day one: redundant inputs, missed inputs,
    prediction history length, misprediction count, interpolation buffer size.
- Do not extract:
  - Godot node/component assumptions, broad entity-spawner framework, or generic
    message serializer hierarchy. Keep the first Odin protocol explicit.

### Heavenlode/Nebula

- Repository: `https://github.com/Heavenlode/Nebula`
- Local clone: `.../multiplayer-prior-art/Nebula`
- Useful files:
  - `addons/Nebula/NEBULA_OVERVIEW.mdc`
  - `addons/Nebula/Core/WorldRunner.cs`
  - `addons/Nebula/Core/NetworkController.cs`
  - `addons/Nebula/Core/NetId.cs`
  - `addons/Nebula/Core/NetProperty.cs`
  - `addons/Nebula/Core/Serialization/NetBuffer.cs`
  - `addons/Nebula/Core/Serialization/NetReader.cs`
  - `addons/Nebula/Core/Serialization/NetWriter.cs`
  - `addons/Nebula/Core/Nodes/NetTransform/NetTransform3D.cs`
- Extract for this project:
  - Separate a top-level network runner from per-world/per-map state ownership.
  - Give networked objects stable ids distinct from local array indices.
  - Track input authority separately from object existence.
  - Use small circular snapshot buffers; Nebula uses an 8-snapshot per-node buffer
    for interpolation.
  - Clear interpolation buffers on teleports or discontinuities.
  - Keep room for interest filtering later, but do not build property-level
    interest before snapshot bandwidth proves it is needed.
  - Explicit serializers/readers/writers are a better fit than reflection or source
    generation for the early Odin implementation.
- Do not extract:
  - Roslyn generators, attribute-driven property sync, editor tooling, or the full
    NetNode abstraction. This project's object model is intentionally simpler.

### nongvantinh/godot-client-server

- Repository: `https://github.com/nongvantinh/godot-client-server`
- Local clone: `.../multiplayer-prior-art/godot-client-server`
- Useful files:
  - `README.md`
  - `core/GameManager.cs`
  - `core/Character.cs`
  - `core/CharacterPredictedData.cs`
  - `core/CharacterInterpolatedData.cs`
- Extract for this project:
  - The smallest useful state structs are very small: input, predicted player
    state, and interpolated render state.
  - Interpolated state can be just position plus orientation at first.
  - The README is a cautionary note about high-level RPC APIs sending too eagerly
    and fighting custom packet layout.
- Do not extract:
  - The later client-trusting MMO direction. This project should keep the server
    authoritative.
  - Per-property RPC replication. Use explicit packet batching instead.

### RomanZhu/Entitas-Sync-Framework

- Repository: `https://github.com/RomanZhu/Entitas-Sync-Framework`
- Local clone: `.../multiplayer-prior-art/Entitas-Sync-Framework`
- Useful files:
  - `README.md`
  - `Assets/Sources/Networking/Client/ClientNetworkSystem.cs`
  - `Assets/Sources/Networking/Server/ServerNetworkSystem.cs`
  - `Assets/Sources/Networking/Server/StateCapture/ServerCaptureCreatedEntitiesSystem.cs`
  - `Assets/Sources/Networking/Server/StateCapture/ServerCaptureRemovedEntitiesSystem.cs`
  - `Assets/Sources/Networking/Server/StateCapture/ServerCreateWorldStateSystem.cs`
  - `Assets/Sources/Networking/Components/ClientDataBuffer.cs`
  - `Assets/Sources/Networking/Components/Sync.cs`
  - `Assets/Sources/Networking/Components/WasSynced.cs`
  - `Assets/Sources/Networking/PackedDataFlags.cs`
- Extract for this project:
  - First packet/world-state bootstrap as a distinct path from incremental changes.
  - Separate created entities, removed entities, changed components/state, and
    per-client data buffers.
  - Bounded/quantized fields are worth considering once packet sizes matter.
  - Client state queues can smooth jitter, but they introduce visible delay.
- Do not extract:
  - Single reliable channel for all traffic. Its README calls out head-of-line
    blocking and low tick-rate constraints.
  - A networking thread and lockless queues for the first pass. Start frame-polled
    and add threading only when measurement says it is needed.
  - ECS code generation. The current fat-struct scene model should remain explicit.

## Patterns To Pull Forward

- Add small explicit structs first: `NetId`, `ClientInputMessage`,
  `ServerSnapshotMessage`, `ReplicatedPlayerState`, and maybe
  `ReplicatedPropState` later.
- Use redundant unreliable input packets with sequence/tick stamps.
- Include snapshot acknowledgements or last-processed input ids in server
  snapshots so the client can discard old inputs.
- Keep interpolation and prediction buffers bounded and inspectable in the debug
  HUD.
- Keep network identity distinct from scene storage identity. `NetId` is the
  protocol/replication id; `ObjectId` is local scene storage only.
- Keep replicated transform buffering in the replication layer so players, props,
  trains, and later movers can share the same sample/interpolation path.
- Start with player-only replication, then add selected dynamic props after packet
  size and CPU costs are visible.
- Follow the Source-style authority split: server-side think/physics owns gameplay
  truth and dynamic prop spawning; client-side think should be for prediction,
  interpolation, audio, particles, and other presentation-only work.
- Replicate dynamic physics props as explicit network state, not by trying to keep
  Box3D deterministic across machines. Start with transform snapshots and add
  awake/sleep, dirty flags, and interpolation history when bandwidth or jitter
  requires it.
- Treat teleports, map reloads, and hot reloads as discontinuities that clear
  interpolation/prediction buffers.
- Keep packet serialization explicit and versioned. Avoid reflection-like generic
  serializers until repeated packet code proves painful.

## First Milestones

1. Transport smoke test.
   - Add a shared ENet wrapper.
   - Add a separate dedicated server entrypoint and CLI build/run commands.
   - Add a combined `net-smoke` command that runs the real game client against
     the dedicated server.
   - Verify connect, disconnect, reliable hello, and clean shutdown.
   - Status: complete for direct ENet localhost smoke with the real app client.

2. Game handshake.
   - Exchange protocol version and map name.
   - Reject map mismatches cleanly.
   - Add a simple map/content identifier before connecting to real gameplay state.
   - Status: protocol version, map-name exchange, content-id exchange, and clean
     rejects are implemented in the smoke protocol.
   - Keep hot reload disabled or session-ending for active multiplayer until a
     deliberate reload policy exists.

3. Authoritative input loop.
   - Client sends local movement/look input with sequence numbers.
   - Server applies input to one authoritative player controller in fixed steps.
   - Server sends player snapshots.
   - Client renders authoritative state without prediction first.
   - Status: the real app can run with `--connect`, complete the handshake, and
     send `User_Cmd` packets built from real player input. The dedicated server
     now owns one shared headless real `Scene`, assigns accepted peers stable
     player ids in `Server_Hello`, resets each player to the map spawn on accept,
     queues deduplicated user commands per session, consumes one queued command
     per player in sequence order on a 64 Hz server tick, and broadcasts full `Server_Snapshot`
     packets at 32 Hz independent of input arrival. Client input packets include recent
     commands so dropped packets can be recovered, and snapshots include the last
      processed command sequence for the receiving client. Player snapshot state
      includes position, velocity, view angles, grounded state, and ground normal
      so reconciliation can reset to the authoritative movement state before
      replaying unacknowledged commands. Clients reject stale or duplicate
      snapshot sequences and render non-camera players from canonical `Scene.players`
      using a small remote interpolation buffer, currently delayed by 100 ms.
      Local-player reconciliation is still pending.

4. Snapshot interpolation.
    - Assign stable network ids to replicated objects.
    - Interpolate remote players and selected dynamic props from snapshot buffers.
    - Bound snapshot size before attempting to replicate every stress-test prop.
   - Status: remote player snapshots are buffered per `ScenePlayer` and rendered
     behind the latest received server tick. Duplicate/stale snapshots are ignored
     by sequence. Dynamic props are still deferred.

5. Local prediction and reconciliation.
   - Predict only the local player controller first.
   - Store local inputs until acknowledged by server snapshots.
   - Correct and replay when error crosses a small threshold.
   - Leave dynamic Suzanne props server-authoritative/interpolated.
   - Status: local player command prediction now stores predicted movement state in
     a fixed command ring. Authoritative snapshots reset the local player to the
     acknowledged server state, replay unacknowledged commands, and expose command,
     ack, prediction error, replay count, and correction count in the debug HUD.
     Dynamic prop rollback is intentionally not implemented; interactions with
     dynamic physics objects can still diverge until those objects become
     server-authoritative replicated state.

6. Server-authoritative dynamic props.
   - Server runs prop spawners and Box3D for Suzanne props.
   - Snapshots replicate a bounded set of Suzanne prop transforms by stable object
     id.
   - Clients upsert those props from snapshots instead of advancing local prop
     physics.
   - Status: Suzanne transform replication is implemented for up to 64 changed
     props per snapshot. Clients buffer prop samples and render them at the same
     delayed snapshot tick used for remote players. Clients also create/update a
     kinematic collision proxy at the latest received authoritative prop transform
     so local player prediction can collide with server-owned props without
     simulating prop rigid-body motion locally. This mirrors the id Tech/Source
     split between prediction collision state and render interpolation, but it is
     still not historical prop rollback. Snapshots carry explicit removed prop ids,
     so absence is not overloaded as deletion. The server keeps per-session known
     prop state and sends new, changed, awake, or periodic-refresh props while
     skipping unchanged sleeping props. `prop_suzanne` supports a `net_policy` key;
     the default `server` policy replicates as authoritative state, while `client`
     marks presentation-only props that are excluded from server snapshots. Prop
     snapshots use `NetId`, not local `ObjectId`, and replicated transform sample
     storage lives in `src/game/replication.odin`. A richer Source-style debris
     policy is still pending.

7. Moving platforms and trains.
   - Source's first useful model is classic `func_train` plus `path_corner`, not
     the richer `func_tracktrain` path-track system.
   - A train should be a server-side kinematic mover, updated in think before
     player movement, with brush render/collision owned by the train object.
   - Minimal map keys should be `func_train.target`, optional `speed`, `wait`,
     `origin`, and `targetname`; `path_corner` should support `targetname`,
     `target`, `origin`, optional `speed`, and optional `wait`.
   - Player riding needs a ground-entity/platform-delta concept in the mover so a
     grounded player inherits train movement for that fixed step.
   - Status: researched against Source SDK 2013. Implementation is pending.

## Known Risks

- Snapshot bandwidth can explode with many dynamic props. Start with player-only
  replication, then add a capped subset of props.
- Quake-style movement prediction must match the server path closely. Avoid
  duplicating movement logic in a separate client-only implementation.
- Hot reload currently preserves game memory but rebuilds Box3D. During active
  multiplayer, that can invalidate network state unless explicitly handled.
- Direct IP connections avoid account/matchmaking work but do not solve NAT.
- Reliable high-rate state packets can cause head-of-line blocking. Keep repeated
  snapshots and input on unreliable sequenced channels.

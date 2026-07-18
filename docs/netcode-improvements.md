# Netcode Improvements

This document tracks the path from the current explicit snapshot protocol to a
more robust Source/s&box-style replication model. Keep the implementation small
and measurable at each step; do not jump straight to a generic networking
framework.

## Current Problem

The plinko map can create many awake dynamic props at once. Around 80 props,
player updates still arrive, but some prop updates appear to freeze.

The current limits make that expected:

- `protocol.MAX_SNAPSHOT_PROPS` is `64`, so one snapshot can carry at most 64 prop
  states.
- `NetServerSession.known_props` is also sized to `MAX_SNAPSHOT_PROPS`, so each
  client can only remember 64 known props total.
- `net_server_add_prop_delta` scans `scene.objects` from the beginning every
  snapshot. If the first 64 props remain awake or dirty, later props are skipped
  every time.
- Player states are packed before prop deltas, so player movement can keep working
  while prop replication starves.

This is a scheduling and packet-budget problem, not a Box3D simulation problem.

## Prior Art

Source and s&box both avoid sending every physics object fully every tick.

- Source uses entity transmit rules, PVS/interest filtering, base entity deltas,
  minimal prop state, awake/sleep state, and client interpolation.
- s&box uses per-connection delta snapshot state, object/component dirty tracking,
  ACK/hash state, visibility filtering, dirty-first ordering, and byte-sized
  unreliable snapshot clusters around MTU scale.
- s&box skips sleeping physics body transforms and smooths client proxy bodies
  toward received authoritative transforms.
- Both designs have explicit overload behavior: low-interest or unchanged objects
  are omitted, and missing updates are recovered by later snapshots rather than by
  reliable high-rate state packets.

## Packet Math

Current packet constants:

- `protocol.MAX_PACKET_SIZE = 4096`
- `protocol.HEADER_SIZE = 9`
- `SERVER_SNAPSHOT_HEADER_PAYLOAD_SIZE = 18`
- `SERVER_PLAYER_STATE_PAYLOAD_SIZE = 49`
- `SERVER_PROP_STATE_PAYLOAD_SIZE = 19`
- `SERVER_REMOVED_PROP_PAYLOAD_SIZE = 4`

Worst-case current snapshot size:

```text
9 + 18 + 32*49 + 64*19 + 64*4 = 3067 bytes
```

With one player and no removed props, a 4096-byte packet could fit about 206 prop
states:

```text
floor((4096 - 9 - 18 - 49) / 19) = 206
```

Do not treat that as the final target. Large UDP datagrams are fragile. A future
s&box-style budget should aim for MTU-sized clusters, roughly `1200` bytes:

```text
floor((1200 - 9 - 18 - 49) / 19) = 59 props with one player
```

The important change is fairness and resending, not simply raising the cap.

## Step 1: Minimal Starvation Fix

Status: complete.

Goal: keep the current protocol and packet shape, but ensure more than 64 active
props eventually update.

Changes:

- Add a server-side known-prop capacity independent from per-packet prop budget,
  for example `NET_SERVER_KNOWN_PROPS :: MAX_OBJECTS` or an explicit smaller cap.
- Change `NetServerSession.known_props` from
  `[protocol.MAX_SNAPSHOT_PROPS]NetServerKnownProp` to
  `[NET_SERVER_KNOWN_PROPS]NetServerKnownProp`.
- Add `NetServerSession.prop_snapshot_cursor: int`.
- Change `net_server_add_prop_delta` to walk replicated props round-robin from the
  session cursor instead of always starting at object index 0.
- Keep `protocol.MAX_SNAPSHOT_PROPS` at `64` as the per-snapshot budget for now.
- Advance the cursor after each snapshot attempt so deferred props are considered
  first on later snapshots.

Implemented notes:

- `NET_SERVER_KNOWN_PROPS` now separates per-client known prop storage from the
  per-packet `protocol.MAX_SNAPSHOT_PROPS` budget.
- `NetServerSession.prop_snapshot_cursor` now tracks where each client should
  resume prop consideration.
- `net_server_add_prop_delta` walks `scene.objects` round-robin from that cursor
  and advances the cursor after sent props.
- When deferred props are sent first and packet budget remains, the scheduler wraps
  and continues filling the snapshot with other send-worthy props.
- Regression tests cover more than one snapshot budget of props and dirty early
  props no longer starving later unknown props.

Expected behavior:

- Under heavy load, each prop may update less often.
- No prop should permanently starve just because earlier props are also awake.
- Existing clients and protocol tests should keep working.

Tests:

- Create more than 64 replicated props in a server/session test.
- Force them to be send-worthy.
- Call `net_server_add_prop_delta` across several snapshots.
- Assert net ids beyond the first 64 eventually appear.
- Assert known prop state can store more than 64 distinct props.

## Step 2: Visibility And Sleep Discipline

Goal: avoid spending snapshot budget on objects that do not need updates.

Changes:

- Treat Box3D sleep/awake state as a first-class replication signal.
- Continue skipping unchanged sleeping props.
- Add periodic refresh for sleeping known props at a low rate.
- Add debug counters for total replicated props, awake props, sent props, deferred
  props, known prop slots used, and snapshot bytes.
- Later, add simple distance or camera/PVS filtering per client before generic
  interest management.

This mirrors Source's transmit rules and s&box visibility filtering without adding
a full visibility system too early.

## Step 3: Byte Budgeting

Status: initial version complete.

Goal: budget snapshots by bytes rather than entity count.

Changes:

- Add protocol helpers for snapshot payload and packet size calculations.
- Add constants such as `NET_SNAPSHOT_TARGET_BYTES :: 1200` and keep
  `protocol.MAX_PACKET_SIZE` as the hard upper bound.
- Pack prop states until adding another state would exceed the target byte budget.
- Keep a hard prop-count limit only as an array safety cap, not as the primary
  scheduling policy.
- Log when state is deferred due to byte budget.

Implemented notes:

- `NET_SERVER_SNAPSHOT_TARGET_BYTES` currently targets `1200` byte snapshot
  packets.
- `net_server_snapshot_prop_budget` computes how many prop states fit after the
  snapshot header, player states, and removed-prop ids already in that packet.
- `protocol.MAX_SNAPSHOT_PROPS` remains the hard array safety cap, but normal prop
  packing now uses the byte target.
- `NetServerSnapshotStats` records per-tick snapshot packet count, byte count, prop
  count, removed-prop count, and whether any client still had deferred props after
  hitting the cluster limit.
- Explicit protocol-level packet size helpers are still pending.

This is the point where local stress maps should become intentionally lossy but
fair: every important prop eventually updates, but not necessarily every snapshot.

## Step 4: Snapshot Clusters

Status: initial version complete.

Goal: send more than one unreliable snapshot cluster when a client has a large
backlog, without creating huge UDP packets.

Changes:

- Split snapshot output into one or more MTU-sized clusters per snapshot tick.
- Give each cluster sequence/part metadata or make each cluster independently
  parseable as a snapshot fragment.
- Keep player state in the first/primary cluster so player movement remains high
  priority.
- Put prop deltas in later clusters ordered by priority and round-robin fairness.

Implemented notes:

- `NET_SERVER_MAX_SNAPSHOT_CLUSTERS` currently allows up to four bounded snapshot
  packets per client per snapshot tick.
- Each cluster is still a normal `Server_Snapshot` packet with its own snapshot
  sequence, so the client can parse it without a new fragment protocol.
- The first cluster carries player state and the client's last processed command.
- Later clusters carry additional prop deltas only.
- The prop round-robin cursor continues across clusters, so large active prop sets
  can use more per-tick bandwidth without one oversized datagram.
- The prop scheduler now reports whether more send-worthy props remain, so the
  server stops without an empty probe cluster when all prop work fit and records
  when the cluster limit deferred more work.
- Explicit cluster part metadata is still deferred until ACK/delta snapshots need
  it.

This copies the useful s&box idea: many changed objects create more bounded
clusters, not one oversized datagram.

## Step 5: Delta State And ACKs

Goal: stop resending fields the client already has, and recover cleanly from lost
unreliable packets.

Changes:

- Add client snapshot acknowledgements.
- Keep per-client known/acked state for replicated objects.
- Separate object create/full-state from transform deltas.
- Once a prop is known, transform-only updates can omit stable fields such as
  `prop_asset_index`.
- Consider per-slot hashes or simple generation numbers before bit-level packing.
- Resend unacknowledged important changes after a timeout.

This is the foundation for Quake/Source/s&box-style robust snapshots.

## Step 6: Compression And Quantization

Status: initial prop transform compression complete.

Goal: reduce bytes per prop after the protocol semantics are proven.

Options:

- Quantize positions to map/local bounds.
- Compress quaternions or send smaller angular representations.
- Use changed-object lists instead of fixed full arrays.
- Split awake/sleep/state flags from transform data.
- Use bit-level writers only after explicit byte writers become the bottleneck.

Implemented notes:

- Prop rotations are still represented as `[4]f32` in game/protocol state, but the
  wire payload now uses smallest-three quaternion compression.
- Prop positions are still represented as `[3]f32` in game/protocol state, but the
  wire payload now quantizes each component to signed 16-bit in a fixed
  `+/-256m` range.
- This changes `SERVER_PROP_STATE_PAYLOAD_SIZE` from the original `34` bytes to
  `19` bytes.
- The fixed position range matches the current map scale assumptions. If larger
  maps need dynamic bounds later, make the quantization range part of map/session
  metadata before shipping those maps.

Do not start here. Compression hides bugs if scheduling, ACKs, and interpolation are
not already solid.

## Step 7: Client Presentation Improvements

Goal: make sparse prop updates look acceptable under load.

Changes:

- Keep rendering from interpolation buffers using continuous fractional render
  time.
- Expose whether each prop/player is interpolating, holding, or extrapolating.
- Add optional short extrapolation for props using last two samples, capped and
  visible in debug counters.
- Smooth client collision proxies deliberately; decide whether local prediction
  should collide against latest authoritative prop transforms or delayed render
  transforms per gameplay need.

## Non-Goals For Now

- Full Box3D deterministic lockstep.
- Full dynamic prop rollback.
- Reliable delivery for high-rate snapshots.
- Generic reflection/property replication.
- Perfectly smooth updates for hundreds of awake props at 32 snapshots/sec without
  culling, prioritization, or reduced fidelity.

## Useful Debug Counters

Add these before or during the scheduling work:

- Snapshot bytes written.
- Prop states sent per snapshot.
- Prop states deferred by budget.
- Total replicated props.
- Awake replicated props.
- Known prop slots used per client.
- Snapshot sequence and last acknowledged snapshot.
- Interpolation buffer sample count per player/prop.
- Frames interpolated, held, and extrapolated.

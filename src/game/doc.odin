/*
package game owns the multiplayer-first game simulation.

The normal SDL client and local loopback path both use client/server semantics.
Clients produce `User_Cmd` input, the authoritative server core steps a scene,
and clients apply snapshots with prediction, reconciliation, and interpolation.

This package owns map loading, scene objects, player movement, entity spawning,
prop collision policy, client prediction, authoritative server sessions, and the
bridge from scene state to renderer-facing draw items.
*/
package game

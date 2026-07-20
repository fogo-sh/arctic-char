/*
package transport wraps Odin's ENet binding behind project-owned types.

The wrapper exposes host creation, client connection, server listening, event
polling, packet sending, and peer handles. It intentionally does not know about
maps, players, snapshots, prediction, or any game entity ids.
*/
package transport

/*
package protocol defines the wire format shared by clients and servers.

It contains packet headers, channel ids, user command payloads, server snapshot
payloads, prop state compression, packet size helpers, and parse/write routines.
It should not know about SDL, ENet hosts, Box3D worlds, or scene ownership.
*/
package protocol

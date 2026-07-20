/*
package main in `src/server` is the headless dedicated server executable.

It owns process setup, command-line parsing, ENet host lifetime, optional network
impairment for local testing, and log routing. The authoritative gameplay and
session logic lives in `game.NetServer`; this package should remain a thin shell
around that core.
*/
package main

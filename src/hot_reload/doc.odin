/*
package main in `src/hot_reload` is the development hot-reload host.

It starts the SDL runtime, loads the game dynamic library from `build/hot_reload`,
resolves exported game callbacks, and swaps libraries during development reloads.
It is a tool for faster iteration, not a second gameplay architecture.
*/
package main

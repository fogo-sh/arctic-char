/*
package engine_sdl owns the native SDL application runtime and SDL GPU renderer.

This is the only supported client application backend. It owns SDL startup,
window events, input polling, wall-clock frame timing, fixed-step callback
dispatch, shader creation, GPU mesh uploads, render targets, and command
recording.

The package depends on generic data from `src/engine`, but game rules stay in
`src/game`. The SDL loop calls game callbacks and renders the frame data returned
by game code.
*/
package engine_sdl

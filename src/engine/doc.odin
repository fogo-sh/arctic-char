/*
package engine contains platform-independent engine types and helpers.

This package is deliberately not the SDL app. It defines shared contracts for
input, launch options, renderer-facing data, mesh loading, qpath filesystem
access, UI commands, renderer math, text data, and generic Box3D helpers.

The SDL runtime lives in `src/engine_sdl`. Game code imports this package for
stable engine-facing types without taking a dependency on SDL-specific command
recording or window lifecycle code.
*/
package engine

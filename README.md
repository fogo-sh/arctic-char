# arctic char*

Small Odin + SDL3 GPU sandbox

Current features:

- SDL3 window and SDL GPU setup
- Metal/MSL on macOS, SPIR-V on Linux-ish platforms, DXIL on Windows
- color-only HLSL shader compiled with `SDL_shadercross`
- GLB mesh loading through Odin's `vendor:cgltf`
- generic renderer mesh handles, depth testing, and 4x MSAA when supported
- Box3D world with fixed-timestep stepping
- static ground collision and 100 falling Suzanne bodies spawned over time

The project is pinned to Odin `dev-2026-07` via `mise.toml`.

## Build

Build and run:

```sh
just build-and-run
```

Other useful commands:

```sh
just build
just build-release
just run
```

`just build` copies `assets/` into `build/assets/` after compiling.

## Shaders

`shaders/hlsl/shader.hlsl` is the source of truth. The generated MSL, SPIR-V,
and DXIL outputs are checked in so the project can build without running the
shader compiler every time.

Regenerate shaders:

```sh
just shaders
```

Check that generated shaders are in sync:

```sh
just check-shaders
```

The `justfile` expects `shadercross` on `PATH`.

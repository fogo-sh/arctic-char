# arctic char*

Tiny Odin + SDL3 GPU sandbox.

Right now this is intentionally just:

- SDL3 window/GPU setup
- `cgltf` loading `assets/suzanne.glb`
- a color-only HLSL shader compiled with `SDL_shadercross`
- a rotating Suzanne render loop
- depth testing and 4x MSAA when supported

Build and run:

```sh
just build-and-run
```

Regenerate shaders:

```sh
just shaders
```

`shaders/hlsl/shader.hlsl` is the source of truth. The generated MSL, SPIR-V,
and DXIL outputs are checked in so the project can build without running the
shader compiler every time.

Check that generated shaders are in sync:

```sh
just check-shaders
```

The repo is pinned to Odin `dev-2026-07` via `mise` for the current baseline.

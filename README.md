# arctic char*

Small Odin + SDL3 GPU sandbox

Current features:

- SDL3 window and SDL GPU setup
- Metal/MSL on macOS, SPIR-V on Linux-ish platforms, DXIL on Windows
- color-only HLSL shader compiled with `SDL_shadercross`
- GLB mesh loading through Odin's `vendor:cgltf`
- generic renderer mesh handles, depth testing, and 4x MSAA when supported
- minimal Clay UI path for SDL GPU rectangles, borders, scissors, and text
- Box3D world with fixed-timestep stepping
- directory-first game filesystem with `--game` search path overrides
- Valve 220 `.map` brush rendering and static map collision
- Quake-style player movement and 100 falling Suzanne bodies spawned over time

The project is pinned to Odin `dev-2026-07` via `mise.toml`.

## Build

Build and run:

```sh
python cli.py build-and-run
```

Other useful commands:

```sh
python cli.py build
python cli.py build-release
python cli.py test
python cli.py macos-app
python cli.py run
```

`python cli.py build` copies `base/` into `build/base/` after compiling. Runtime content
is loaded through qpaths under `base` by default.

`python cli.py test` runs Odin unit tests for pure package logic. The smoke test
is separate: it launches the real app for a fixed duration and catches startup or
runtime crashes, but it does not assert gameplay state by itself.

Run a different game directory before falling back to `base`:

```sh
python cli.py run -- --game mymod
```

Load a different map qpath by name:

```sh
python cli.py run -- --map test
```

## Xcode Metal Debugger

On macOS, open the launcher project:

```sh
open xcode/ArcticCharMetal.xcodeproj
```

The shared `ArcticCharMetal` scheme builds `build/ArcticChar.app` with
`python cli.py macos-app`, launches it with the repo root as the working
directory, passes `--fullscreen`, and enables Xcode GPU Frame Capture for Metal.
The bundle also declares `LSApplicationCategoryType=public.app-category.games`
and `LSSupportsGameMode=true`; macOS Game Mode can activate when the app is the
frontmost fullscreen game. Press Run in Xcode, then use the Metal capture button
in the debug bar.

You can also build the bundle manually:

```sh
python cli.py macos-app
```

## Shaders

`shaders/hlsl/*.hlsl` files are the source of truth. The generated MSL, SPIR-V,
and DXIL outputs are checked in so the project can build without running the
shader compiler every time.

Regenerate shaders:

```sh
python cli.py shaders
```

Check that generated shaders are in sync:

```sh
python cli.py check-shaders
```

The CLI expects `shadercross` on `PATH`.

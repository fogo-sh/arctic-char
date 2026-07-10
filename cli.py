#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
BASE = ROOT / "base"
APP_NAME = "arctic-char"
HOT_DIR = BUILD / "hot_reload"
TRENCHBROOM_PROFILE_DIR = ROOT / "tools" / "trenchbroom" / "ArcticChar"
CLAY_DIR = ROOT / "vendor" / "clay"
SHADER_OUTPUTS = [
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "SPIRV", "vertex", "VertexMain", ROOT / "shaders" / "spv" / "shader.spv.vert"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "SPIRV", "fragment", "FragmentMain", ROOT / "shaders" / "spv" / "shader.spv.frag"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "SPIRV", "vertex", "SkyVertexMain", ROOT / "shaders" / "spv" / "sky.spv.vert"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "SPIRV", "fragment", "SkyFragmentMain", ROOT / "shaders" / "spv" / "sky.spv.frag"),
    (ROOT / "shaders" / "hlsl" / "ui.hlsl", "SPIRV", "vertex", "UiVertexMain", ROOT / "shaders" / "spv" / "ui.spv.vert"),
    (ROOT / "shaders" / "hlsl" / "ui.hlsl", "SPIRV", "fragment", "UiFragmentMain", ROOT / "shaders" / "spv" / "ui.spv.frag"),
    (ROOT / "shaders" / "hlsl" / "text.hlsl", "SPIRV", "vertex", "TextVertexMain", ROOT / "shaders" / "spv" / "text.spv.vert"),
    (ROOT / "shaders" / "hlsl" / "text.hlsl", "SPIRV", "fragment", "TextFragmentMain", ROOT / "shaders" / "spv" / "text.spv.frag"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "MSL", "vertex", "VertexMain", ROOT / "shaders" / "msl" / "shader.msl.vert"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "MSL", "fragment", "FragmentMain", ROOT / "shaders" / "msl" / "shader.msl.frag"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "MSL", "vertex", "SkyVertexMain", ROOT / "shaders" / "msl" / "sky.msl.vert"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "MSL", "fragment", "SkyFragmentMain", ROOT / "shaders" / "msl" / "sky.msl.frag"),
    (ROOT / "shaders" / "hlsl" / "ui.hlsl", "MSL", "vertex", "UiVertexMain", ROOT / "shaders" / "msl" / "ui.msl.vert"),
    (ROOT / "shaders" / "hlsl" / "ui.hlsl", "MSL", "fragment", "UiFragmentMain", ROOT / "shaders" / "msl" / "ui.msl.frag"),
    (ROOT / "shaders" / "hlsl" / "text.hlsl", "MSL", "vertex", "TextVertexMain", ROOT / "shaders" / "msl" / "text.msl.vert"),
    (ROOT / "shaders" / "hlsl" / "text.hlsl", "MSL", "fragment", "TextFragmentMain", ROOT / "shaders" / "msl" / "text.msl.frag"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "DXIL", "vertex", "VertexMain", ROOT / "shaders" / "dxil" / "shader.dxil.vert"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "DXIL", "fragment", "FragmentMain", ROOT / "shaders" / "dxil" / "shader.dxil.frag"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "DXIL", "vertex", "SkyVertexMain", ROOT / "shaders" / "dxil" / "sky.dxil.vert"),
    (ROOT / "shaders" / "hlsl" / "shader.hlsl", "DXIL", "fragment", "SkyFragmentMain", ROOT / "shaders" / "dxil" / "sky.dxil.frag"),
    (ROOT / "shaders" / "hlsl" / "ui.hlsl", "DXIL", "vertex", "UiVertexMain", ROOT / "shaders" / "dxil" / "ui.dxil.vert"),
    (ROOT / "shaders" / "hlsl" / "ui.hlsl", "DXIL", "fragment", "UiFragmentMain", ROOT / "shaders" / "dxil" / "ui.dxil.frag"),
    (ROOT / "shaders" / "hlsl" / "text.hlsl", "DXIL", "vertex", "TextVertexMain", ROOT / "shaders" / "dxil" / "text.dxil.vert"),
    (ROOT / "shaders" / "hlsl" / "text.hlsl", "DXIL", "fragment", "TextFragmentMain", ROOT / "shaders" / "dxil" / "text.dxil.frag"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Arctic Char project tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("clean", help="Remove build outputs")
    sub.add_parser("build", help="Debug build and copy base assets")
    sub.add_parser("build-release", help="Release build and copy base assets")
    run_parser = sub.add_parser("run", help="Run the normal executable")
    run_parser.add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("build-and-run", help="Build then run")
    sub.add_parser("hot-game", help="Build hot-reload game dynamic library")
    sub.add_parser("hot-host", help="Build hot-reload host")
    sub.add_parser("hot-build", help="Build hot-reload game and host, then copy assets")
    hot_run_parser = sub.add_parser("hot-run", help="Build and run hot-reload host")
    hot_run_parser.add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("collision-mesh", help="Regenerate Suzanne collision mesh with Blender")
    sub.add_parser("clay-lib", help="Build vendored Clay static library for Odin")
    sub.add_parser("shaders", help="Regenerate checked-in shaders")
    sub.add_parser("check-shaders", help="Verify generated shaders are in sync")
    sub.add_parser("trenchbroom-profile", help="Generate TrenchBroom GameConfig and FGD")
    sub.add_parser("trenchbroom-install", help="Install generated TrenchBroom profile")
    smoke_parser = sub.add_parser("smoke", help="Run a short smoke test")
    smoke_parser.add_argument("--seconds", type=float, default=5.0)

    args = parser.parse_args()
    commands = {
        "clean": cmd_clean,
        "build": cmd_build,
        "build-release": cmd_build_release,
        "run": lambda: cmd_run(args.args),
        "build-and-run": lambda: (cmd_build(), cmd_run([])),
        "hot-game": cmd_hot_game,
        "hot-host": cmd_hot_host,
        "hot-build": cmd_hot_build,
        "hot-run": lambda: (cmd_hot_build(), cmd_hot_run(args.args)),
        "collision-mesh": cmd_collision_mesh,
        "clay-lib": cmd_clay_lib,
        "shaders": cmd_shaders,
        "check-shaders": cmd_check_shaders,
        "trenchbroom-profile": cmd_trenchbroom_profile,
        "trenchbroom-install": cmd_trenchbroom_install,
        "smoke": lambda: cmd_smoke(args.seconds),
    }
    commands[args.command]()
    return 0


def run(cmd: list[str | Path], *, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def exe_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def dylib_ext() -> str:
    system = platform.system()
    if system == "Windows":
        return ".dll"
    if system == "Darwin":
        return ".dylib"
    return ".so"


def app_path() -> Path:
    return BUILD / f"{APP_NAME}{exe_suffix()}"


def hot_host_path() -> Path:
    return HOT_DIR / f"{APP_NAME}-hot{exe_suffix()}"


def copy_base_to(dest: Path) -> None:
    target = dest / "base"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(BASE, target)


def cmd_clean() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)


def cmd_build() -> None:
    cmd_clean()
    cmd_clay_lib()
    BUILD.mkdir(parents=True, exist_ok=True)
    run(["odin", "build", ".", "-debug", f"-out:{app_path()}"])
    copy_base_to(BUILD)


def cmd_build_release() -> None:
    cmd_clean()
    cmd_clay_lib()
    BUILD.mkdir(parents=True, exist_ok=True)
    run(["odin", "build", ".", f"-out:{app_path()}"])
    copy_base_to(BUILD)


def cmd_run(extra_args: list[str]) -> None:
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    run([app_path(), *extra_args])


def cmd_hot_game() -> None:
    cmd_clay_lib()
    HOT_DIR.mkdir(parents=True, exist_ok=True)
    ext = dylib_ext()
    tmp = HOT_DIR / f"game_tmp{ext}"
    final = HOT_DIR / f"game{ext}"
    run(["odin", "build", "src/game", "-debug", "-build-mode:dll", f"-out:{tmp}"])
    os.replace(tmp, final)


def cmd_hot_host() -> None:
    HOT_DIR.mkdir(parents=True, exist_ok=True)
    run(["odin", "build", "src/hot_reload", "-debug", f"-out:{hot_host_path()}"])


def cmd_hot_build() -> None:
    cmd_hot_game()
    cmd_hot_host()
    copy_base_to(HOT_DIR)


def cmd_hot_run(extra_args: list[str]) -> None:
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    run([hot_host_path(), *extra_args])


def cmd_collision_mesh() -> None:
    blender = os.environ.get("BLENDER", "blender")
    try:
        subprocess.run([blender, "--version"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit("Blender not found. Install Blender or set BLENDER=/path/to/blender.")
    run([
        blender,
        "--background",
        "--python",
        ROOT / "tools" / "make_collision_mesh.py",
        "--",
        BASE / "models" / "suzanne.glb",
        BASE / "models" / "suzanne_collision.glb",
        "48",
    ])


def clay_library_path() -> Path:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return BUILD / "clay" / ("macos-arm64" if machine == "arm64" else "macos") / "clay.a"
    if system == "Linux":
        return BUILD / "clay" / "linux" / "clay.a"
    if system == "Windows":
        return BUILD / "clay" / "windows" / "clay.lib"
    raise SystemExit(f"Unsupported platform for Clay static library: {system}")


def cmd_clay_lib() -> None:
    clay_h = CLAY_DIR / "clay.h"
    if not clay_h.exists():
        raise SystemExit(f"Missing vendored Clay header: {clay_h}")

    BUILD.mkdir(parents=True, exist_ok=True)
    output = clay_library_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    obj = BUILD / "clay.o"
    source = BUILD / "clay.c"
    shutil.copyfile(clay_h, source)

    system = platform.system()
    if system == "Windows":
        run([
            "clang",
            "-c",
            "-DCLAY_IMPLEMENTATION",
            "-o",
            output,
            "-ffreestanding",
            "-target",
            "x86_64-pc-windows-msvc",
            "-fuse-ld=llvm-lib",
            "-static",
            "-O3",
            source,
        ])
        return

    compile_cmd = ["clang", "-c", "-DCLAY_IMPLEMENTATION", "-o", obj, "-ffreestanding", "-static", "-fPIC", "-O3", source]
    if system == "Darwin" and platform.machine().lower() != "arm64":
        compile_cmd[5:5] = ["-target", "x86_64-apple-darwin"]
    elif system == "Linux":
        compile_cmd[5:5] = ["-target", "x86_64-unknown-linux-gnu"]
    run(compile_cmd)
    if output.exists():
        output.unlink()
    run(["ar", "r", output, obj])


def cmd_shaders() -> None:
	for _, _, _, _, output in SHADER_OUTPUTS:
		output.parent.mkdir(parents=True, exist_ok=True)
	for source, dest, stage, entry, output in SHADER_OUTPUTS:
		run(["shadercross", source, "-s", "HLSL", "-d", dest, "-t", stage, "-e", entry, "-o", output])
		normalize_shader_output(output)


def normalize_shader_output(path: Path) -> None:
	if path.suffix not in {".vert", ".frag"} or path.parent.name != "msl":
		return
	lines = path.read_text(encoding="utf-8").splitlines()
	while lines and lines[-1] == "":
		lines.pop()
	path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def shader_outputs() -> list[Path]:
	return [output for _, _, _, _, output in SHADER_OUTPUTS]


def file_hashes(paths: list[Path]) -> dict[Path, str]:
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def cmd_check_shaders() -> None:
    paths = shader_outputs()
    before = file_hashes(paths)
    cmd_shaders()
    after = file_hashes(paths)
    if before != after:
        changed = [str(path.relative_to(ROOT)) for path in paths if before[path] != after[path]]
        raise SystemExit("Generated shaders are out of sync: " + ", ".join(changed))


def cmd_smoke(seconds: float) -> None:
    process = subprocess.Popen([str(app_path())], cwd=ROOT)
    try:
        time.sleep(seconds)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def cmd_trenchbroom_profile() -> None:
    write_trenchbroom_profile(TRENCHBROOM_PROFILE_DIR)
    print(f"Generated TrenchBroom profile: {TRENCHBROOM_PROFILE_DIR.relative_to(ROOT)}")


def cmd_trenchbroom_install() -> None:
    cmd_trenchbroom_profile()
    dest = trenchbroom_install_dir()
    write_trenchbroom_profile(dest)
    print(f"Installed TrenchBroom profile to: {dest}")


def trenchbroom_install_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "TrenchBroom" / "games" / "ArcticChar"
    if system == "Windows":
        appdata = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if not appdata:
            raise SystemExit("APPDATA and LOCALAPPDATA are not set; cannot locate TrenchBroom user data directory")
        return Path(appdata) / "TrenchBroom" / "games" / "ArcticChar"
    if system in {"Linux", "FreeBSD"}:
        return Path.home() / ".TrenchBroom" / "games" / "ArcticChar"
    return Path.home() / ".TrenchBroom" / "games" / "ArcticChar"


def write_trenchbroom_profile(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "GameConfig.cfg").write_text(generate_game_config(), encoding="utf-8")
    (out_dir / "ArcticChar.fgd").write_text(generate_fgd(), encoding="utf-8")


def generate_game_config() -> str:
    return """{
    "version": 9,
    "name": "Arctic Char",
    "fileformats": [
        { "format": "Valve" }
    ],
    "filesystem": {
        "searchpath": ".",
        "packageformat": { "extension": ".pak", "format": "idpak" }
    },
    "materials": {
        "root": "textures",
        "format": { "extensions": [".png", ".jpg", ".jpeg", ".tga"], "format": "image" }
    },
    "entities": {
        "definitions": [ "ArcticChar.fgd" ],
        "defaultcolor": "0.6 0.6 0.6 1.0"
    },
    "tags": {
        "brush": [
            {
                "name": "Trigger",
                "attribs": [ "transparent" ],
                "match": "classname",
                "pattern": "trigger*",
                "material": "trigger"
            }
        ],
        "brushface": []
    },
    "faceattribs": {
        "surfaceflags": [],
        "contentflags": []
    },
    "softMapBounds": "-4096 -4096 -4096 4096 4096 4096"
}
"""


def generate_fgd() -> str:
    lines = [
        "// Generated by cli.py from src/game/entity_definitions.odin",
        "// Do not edit this file by hand; edit the Odin metadata and regenerate.",
        "",
    ]
    for entity in load_entity_definitions():
        lines.extend(fgd_entity_lines(entity))
        lines.append("")
    return "\n".join(lines)


def fgd_entity_lines(entity: dict[str, object]) -> list[str]:
    kind = "@SolidClass" if entity["editor_kind"] == "Solid" else "@PointClass"
    parts = [kind]
    if entity["editor_kind"] == "Point":
        parts.append(f"size({vec3(entity['size_min'])}, {vec3(entity['size_max'])})")
    parts.append(f"color({vec3(entity['color'])})")
    header = f"{' '.join(parts)} = {entity['classname']} : \"{entity['description']}\""
    properties = entity["properties"]
    if not properties:
        return [header + " []"]
    lines = [header, "["]
    for prop in properties:  # type: ignore[assignment]
        line = f"    {prop['name']}({fgd_property_type(prop['type'])}) : \"{prop['label']}\""
        default = prop.get("default_value", "")
        description = prop.get("description", "")
        if default:
            line += f" : {default}"
        elif description:
            line += " : \"\""
        if description:
            line += f" : \"{description}\""
        lines.append(line)
    lines.append("]")
    return lines


def vec3(value: object) -> str:
    if isinstance(value, dict):
        x, y, z = value["x"], value["y"], value["z"]
        return f"{x:g} {y:g} {z:g}"
    x, y, z = value  # type: ignore[misc]
    return f"{x:g} {y:g} {z:g}"


def fgd_property_type(kind: str) -> str:
    return {
        "String": "string",
        "Integer": "integer",
        "Float": "float",
        "TargetSource": "target_source",
        "TargetDestination": "target_destination",
    }[kind]


def load_entity_definitions() -> list[dict[str, object]]:
    result = subprocess.run(
        ["odin", "run", "tools/entity_metadata"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(result.stdout)


if __name__ == "__main__":
    raise SystemExit(main())

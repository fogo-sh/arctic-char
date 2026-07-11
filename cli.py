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
import signal
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
BASE = ROOT / "base"
APP_NAME = "arctic-char"
SERVER_NAME = "arctic-char-server"
MACOS_BUNDLE_NAME = "ArcticChar"
HOT_DIR = BUILD / "hot_reload"
ODIN_TEST_PACKAGES = ["src/engine", "src/game", "src/protocol"]
DEFAULT_NET_PORT = "29001"
DEFAULT_NET_MAP = "test"
DEFAULT_NET_CONTENT_ID = "0"
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
    sub.add_parser("test", help="Run Odin unit tests")
    sub.add_parser("macos-app", help="Build a macOS .app bundle for Xcode Metal capture")
    run_parser = sub.add_parser("run", help="Run the normal executable")
    run_parser.add_argument("args", nargs=argparse.REMAINDER)
    build_run_parser = sub.add_parser("build-and-run", help="Build then run the app")
    build_run_parser.add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("server-build", help="Build the dedicated server")
    server_run_parser = sub.add_parser("server-run", help="Run the dedicated server")
    server_run_parser.add_argument("args", nargs=argparse.REMAINDER)
    server_smoke_parser = sub.add_parser("server-smoke", help="Run the dedicated server briefly")
    server_smoke_parser.add_argument("--seconds", type=float, default=2.0)
    net_smoke_parser = sub.add_parser("net-smoke", help="Run dedicated server and real game client smoke")
    net_smoke_parser.add_argument("--seconds", type=float, default=5.0)
    net_smoke_parser.add_argument("--port", type=int, default=29001)
    net_smoke_parser.add_argument("--map", default="test")
    net_smoke_parser.add_argument("--content-id", type=int, default=0)
    mp_test_parser = sub.add_parser("mp-test", help="Run one local server and two real clients")
    mp_test_parser.add_argument("--seconds", type=float, default=0.0)
    mp_test_parser.add_argument("--port", default=DEFAULT_NET_PORT)
    mp_test_parser.add_argument("--map", default=DEFAULT_NET_MAP)
    mp_test_parser.add_argument("--content-id", default=DEFAULT_NET_CONTENT_ID)
    mp_test_parser.add_argument("--no-build", action="store_true")
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
        "test": cmd_test,
        "macos-app": cmd_macos_app,
        "run": lambda: cmd_run(args.args),
        "build-and-run": lambda: cmd_build_and_run(args.args),
        "server-build": cmd_server_build,
        "server-run": lambda: cmd_server_run(args.args),
        "server-smoke": lambda: cmd_server_smoke(args.seconds),
        "net-smoke": lambda: cmd_net_smoke(args.seconds, args.port, args.map, args.content_id),
        "mp-test": lambda: cmd_mp_test(args.seconds, args.port, args.map, args.content_id, args.no_build),
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


def server_path() -> Path:
    return BUILD / f"{SERVER_NAME}{exe_suffix()}"


def macos_app_path() -> Path:
    return BUILD / f"{MACOS_BUNDLE_NAME}.app"


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
    build_app_debug()
    build_server_debug()
    copy_base_to(BUILD)


def cmd_build_release() -> None:
    cmd_clean()
    cmd_clay_lib()
    BUILD.mkdir(parents=True, exist_ok=True)
    run(["odin", "build", ".", f"-out:{app_path()}"])
    copy_base_to(BUILD)


def build_app_debug() -> None:
    run(["odin", "build", ".", "-debug", f"-out:{app_path()}"])


def build_server_debug() -> None:
    require_system_enet()
    run(["odin", "build", "src/server", "-debug", f"-out:{server_path()}"])


def cmd_build_and_run(extra_args: list[str]) -> None:
    cmd_build()
    cmd_run(extra_args)


def cmd_test() -> None:
    for package in ODIN_TEST_PACKAGES:
        run(["odin", "test", package, "-debug"])


def cmd_macos_app() -> None:
    if platform.system() != "Darwin":
        raise SystemExit("macos-app is only supported on macOS")
    cmd_build()
    bundle = macos_app_path()
    contents = bundle / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    shutil.copy2(app_path(), macos / APP_NAME)
    shutil.copytree(BUILD / "base", resources / "base", dirs_exist_ok=True)
    (contents / "Info.plist").write_text(macos_info_plist(), encoding="utf-8")
    run(["plutil", "-lint", contents / "Info.plist"])
    run(["chmod", "+x", macos / APP_NAME])
    run(["codesign", "-fs", "-", bundle])
    print(f"Built macOS app bundle: {bundle.relative_to(ROOT)}")


def macos_info_plist() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>{MACOS_BUNDLE_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>Arctic Char</string>
    <key>CFBundleIdentifier</key>
    <string>sh.fogo.arctic-char.dev</string>
    <key>CFBundleExecutable</key>
    <string>{APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0-dev</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppSleepDisabled</key>
    <true/>
    <key>METAL_DEVICE_WRAPPER_TYPE</key>
    <integer>1</integer>
    <key>MTL_ENABLE_GPU_CAPTURE</key>
    <true/>
    <key>LSEnvironment</key>
    <dict>
        <key>MTL_CAPTURE_ENABLED</key>
        <string>1</string>
        <key>MTL_DEBUG_LAYER</key>
        <string>1</string>
        <key>MTL_ENABLE_DEBUG_INFO</key>
        <string>1</string>
        <key>MTL_HUD_ENABLED</key>
        <string>1</string>
    </dict>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.games</string>
    <key>LSSupportsGameMode</key>
    <true/>
    <key>MTLDevicePreference</key>
    <string>HighPerformance</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
</dict>
</plist>
"""


def cmd_run(extra_args: list[str]) -> None:
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    run([app_path(), *extra_args])


def cmd_server_build() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    build_server_debug()


def cmd_server_run(extra_args: list[str]) -> None:
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    run([server_path(), *extra_args])


def cmd_server_smoke(seconds: float) -> None:
    cmd_server_build()
    run([server_path(), "--seconds", str(seconds)])


def cmd_net_smoke(seconds: float, port: int, map_name: str, content_id: int) -> None:
    cmd_build()

    server = subprocess.Popen(
        [str(server_path()), "--port", str(port), "--map", map_name, "--content-id", str(content_id)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    client = None
    try:
        time.sleep(0.2)
        if server.poll() is not None:
            server_output = collect_process_output(server)
            raise SystemExit(f"net-smoke server exited early with {server.returncode}\n{server_output}")
        client = subprocess.Popen([
            str(app_path()),
            "--connect", "127.0.0.1",
            "--port", str(port),
            "--map", map_name,
            "--content-id", str(content_id),
        ], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise SystemExit(f"net-smoke server exited early with {server.returncode}")
            if client.poll() not in (None, 0):
                raise SystemExit(f"net-smoke client exited early with {client.returncode}")
            time.sleep(0.05)
    finally:
        client_output = terminate_process_and_collect(client)
        server_output = terminate_process_and_collect(server)

    print(server_output, end="")
    print(client_output, end="")
    require_output_marker(server_output, "Client hello")
    require_output_marker(server_output, "User cmd")
    require_output_marker(client_output, "Server accepted network session")


def cmd_mp_test(seconds: float, port: str, map_name: str, content_id: str, no_build: bool) -> None:
    if not no_build:
        cmd_build()

    common = ["--connect", "127.0.0.1", "--port", port, "--map", map_name, "--content-id", content_id]
    processes: list[subprocess.Popen] = []

    def request_shutdown(signum, frame):
        raise KeyboardInterrupt

    old_sigint = signal.signal(signal.SIGINT, request_shutdown)
    old_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    try:
        server = spawn_process([server_path(), "--port", port, "--map", map_name, "--content-id", content_id])
        processes.append(server)
        time.sleep(0.35)

        client_a = spawn_process([app_path(), *common])
        processes.append(client_a)
        time.sleep(0.5)

        client_b = spawn_process([app_path(), *common])
        processes.append(client_b)

        print("mp-test running: one server and two clients. Press Ctrl-C to stop.")
        start = time.monotonic()
        while True:
            for process in processes:
                if process.poll() not in (None, 0):
                    raise subprocess.CalledProcessError(process.returncode, process.args)
            if seconds > 0 and time.monotonic() - start >= seconds:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        for process in reversed(processes):
            terminate_process(process)


def spawn_process(cmd: list[str | Path]) -> subprocess.Popen:
    print("+ " + " ".join(str(part) for part in cmd))
    return subprocess.Popen([str(part) for part in cmd], cwd=ROOT)


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def require_output_marker(output: str, marker: str) -> None:
    if marker not in output:
        raise SystemExit(f"Expected smoke output marker not found: {marker}")


def collect_process_output(process: subprocess.Popen) -> str:
    output, _ = process.communicate(timeout=5)
    return output or ""


def terminate_process_and_collect(process: subprocess.Popen | None) -> str:
    if process is None:
        return ""
    if process.poll() is None:
        process.terminate()
    try:
        output, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate(timeout=5)
    return output or ""


def require_system_enet() -> None:
    candidates = [
        Path("/opt/homebrew/lib/libenet.dylib"),
        Path("/opt/homebrew/lib/libenet.a"),
        Path("/usr/local/lib/libenet.dylib"),
        Path("/usr/local/lib/libenet.a"),
        Path("/usr/lib/libenet.dylib"),
        Path("/usr/lib/libenet.a"),
    ]
    if any(path.exists() for path in candidates):
        return

    system = platform.system()
    if system == "Darwin":
        hint = "Install it with `brew install enet`."
    elif system == "Linux":
        hint = "Install the ENet development package, e.g. `libenet-dev` or `enet-devel`."
    else:
        hint = "Install ENet and make sure the linker can find it."
    raise SystemExit(f"Missing system ENet library required by Odin's vendor:ENet binding. {hint}")


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
        "5",
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
    cmd_build()
    process = subprocess.Popen([str(app_path())], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(seconds)
    finally:
        output = terminate_process_and_collect(process)
    print(output, end="")
    require_output_marker(output, "Started local loopback client/server session")


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

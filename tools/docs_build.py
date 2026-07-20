#!/usr/bin/env python3

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DOCS_OUT = BUILD / "docs"
DOCS_VIEWER = ROOT / "tools" / "docs_viewer"
DOCS_ENTRYPOINT = ROOT / "tools" / "odin_doc_all" / "all.odin"
DOCS_CONFIG = ROOT / "docs" / "odin-doc-config.json"

def executable_name(name: str) -> str:
    return f"{name}.exe" if __import__("os").name == "nt" else name


DOCS_JSON_BIN = BUILD / "tools" / executable_name("odin-doc-json")


def main() -> int:
    if DOCS_OUT.exists():
        shutil.rmtree(DOCS_OUT)
    DOCS_OUT.mkdir(parents=True, exist_ok=True)

    build_json_exporter()
    doc_base = DOCS_OUT / "arctic-char"
    run(["odin", "doc", DOCS_ENTRYPOINT, "-file", "-all-packages", "-doc-format", f"-out:{doc_base}"])
    run([DOCS_JSON_BIN, doc_base.with_suffix(".odin-doc"), DOCS_CONFIG, DOCS_OUT / "docs.json"], cwd=DOCS_OUT)
    copy_viewer()

    print(f"Generated docs viewer: {(DOCS_OUT / 'index.html').relative_to(ROOT)}")
    return 0


def build_json_exporter() -> None:
    BUILD.joinpath("tools").mkdir(parents=True, exist_ok=True)
    run(["odin", "build", "tools/odin_doc_json", f"-out:{DOCS_JSON_BIN}"])


def copy_viewer() -> None:
    for name in ["index.html", "style.css", "viewer.js"]:
        shutil.copy2(DOCS_VIEWER / name, DOCS_OUT / name)


def run(cmd: list[str | Path], *, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())

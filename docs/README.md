# Documentation Hub

This directory holds curated explanations. Generated HTML API docs live under
`build/docs` after running `python cli.py odin-doc`.

Start here:

- [Runtime Flow](runtime-flow.md): how control moves from executable entrypoints
  through SDL, game callbacks, scene simulation, networking, and rendering.
- [System Map](system-map.md): short descriptions of the major packages and files
  with source links.
- [Multiplayer](multiplayer.md): server-authoritative networking direction and
  prior-art notes.
- [Engine Decisions](engine-decisions.md): cross-cutting choices that should stay
  explicit.
- [Netcode Improvements](netcode-improvements.md): staged packet/snapshot work and
  packet math.
- [Source 2 Data Notes](source2-data-notes.md): notes from Source 2 data research.

Generated docs:

- Run `python cli.py odin-doc`.
- Open `build/docs/index.html` in a browser.
- The generated site uses Odin's `.odin-doc` output as source data, converts it
  to `docs.json`, then renders that JSON with the local vanilla JS viewer.

Source-link convention:

- Link to files or stable declarations rather than fragile line numbers unless a
  line number is important for a review.
- Prefer links like `../src/game/net_server.odin` when explaining ownership.
- Keep long narrative in Markdown and package/API summaries in `doc.odin` comments.

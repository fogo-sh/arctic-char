package odin_doc_all

// Imports every first-party package that should appear in generated docs.
// This mirrors Odin's own examples/all documentation entrypoint.

@(require) import "../../src/engine"
@(require) import "../../src/engine_sdl"
@(require) import "../../src/game"
@(require) import "../../src/protocol"
@(require) import "../../src/net"
@(require) import "../../src/server"

main :: proc() {}

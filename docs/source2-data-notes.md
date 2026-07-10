# Source 2 Data Format Notes

Source 2 is useful as a design reference because it separates editable source
assets from compiled runtime resources.

## Key Takeaways

- Source 2 uses human-authored source files such as `.vmap`, `.vmat`, `.vmdl`,
  `.vpcf`, `.vdata`, and `.vents`.
- Compiled runtime files use a `_c` suffix, such as `.vmap_c`, `.vmat_c`, and
  `.vmdl_c`.
- ValveResourceFormat documents compiled Source 2 files as little-endian,
  block-based resource containers.
- The compiled resource header records file size, header version, resource
  version, block table offset, and block count.
- Each block table entry stores a FourCC block type, relative block offset, and
  block size.
- Common block roles include metadata/edit info, dependencies, external refs,
  and primary typed data.
- Resource type can be inferred from extension, compiler identifiers in edit
  info, or input dependencies.
- KeyValues3 is the modern structured text/binary data format: JSON-like, but
  with richer scalar typing, arrays, binary blobs, flags, annotations, and
  versioned/binary encodings.
- ValveResourceFormat and ValveKeyValue are reverse-engineered community
  references, not official Valve specifications.

## Design Implications For Arctic Char

- Keep editable source files simple and text-first for now. Valve 220 `.map`
  plus generated FGD is a good starting point.
- Do not make the editor format the runtime format. Plan for a later compile
  step that turns source maps/assets into a compact runtime package.
- Prefer one canonical schema for entity classes and asset metadata, then
  generate editor/runtime helpers from it.
- Consider a future KV3-like text format for richer authored data that no
  longer fits `.map` properties cleanly: entity archetypes, physics materials,
  scripted sequences, particle definitions, and per-map metadata.
- If binary assets become necessary, use a block-container design rather than
  one giant serialized struct. Blocks make partial loading, version migration,
  dependency inspection, and debugging easier.
- Keep compiled artifacts disposable. Source files should remain the reviewable
  authority; compiled files can be regenerated.

## Possible Future Layout

```text
base/
  maps/test.map              # editor-authored brush/entity source
  maps/test.level.kv3        # optional richer map metadata source
  models/suzanne.glb         # source-ish imported model
  materials/*.achar_mat      # text material source

build/base/
  maps/test.map              # current direct-load path
  compiled/test.level_c      # future block container
```

## References

- Source 2 Viewer / ValveResourceFormat resource guide:
  `https://s2v.app/ValveResourceFormat/guides/read-resource.html`
- ValveResourceFormat repository:
  `https://github.com/ValveResourceFormat/ValveResourceFormat`
- ValveKeyValue repository:
  `https://github.com/ValveResourceFormat/ValveKeyValue`
- KeyValues3 overview, when accessible:
  `https://developer.valvesoftware.com/wiki/KeyValues3`

#!/usr/bin/env python3
import sys
from pathlib import Path

import bpy


def argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def mesh_vertex_count(obj: bpy.types.Object) -> int:
    return len(obj.data.vertices)


def rebuild_convex_hull(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.convex_hull(delete_unused=True, use_existing_faces=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def main() -> int:
    args = argv_after_double_dash()
    if len(args) != 3:
        print("usage: blender --background --python tools/make_collision_mesh.py -- INPUT.glb OUTPUT.glb TARGET_VERTICES")
        return 2

    input_path = Path(args[0]).resolve()
    output_path = Path(args[1]).resolve()
    target_vertices = int(args[2])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.import_scene.gltf(filepath=str(input_path))
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        print(f"no mesh objects found in {input_path}")
        return 1

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = "SuzanneCollision"
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    rebuild_convex_hull(obj)
    for _ in range(8):
        if mesh_vertex_count(obj) <= target_vertices:
            break
        modifier = obj.modifiers.new("collision_decimate", "DECIMATE")
        modifier.ratio = max(0.01, 0.8 * target_vertices / mesh_vertex_count(obj))
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        rebuild_convex_hull(obj)

    if mesh_vertex_count(obj) > target_vertices:
        print(f"could not reduce convex hull below {target_vertices} vertices; got {mesh_vertex_count(obj)}")
        return 1

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(output_path), export_format="GLB", use_selection=True)

    print(f"wrote {output_path} with {mesh_vertex_count(obj)} vertices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

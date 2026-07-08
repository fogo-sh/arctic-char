set windows-powershell

clean:
  rm -rf ./build

build: clean
  mkdir -p build
  odin build . -debug -out:./build/arctic-char
  cp -r ./assets ./build/assets

build-release: clean
  mkdir -p build
  odin build . -out:./build/arctic-char
  cp -r ./assets ./build/assets

run:
  ./build/arctic-char

build-and-run: build run

collision-mesh:
  #!/usr/bin/env sh
  BLENDER="${BLENDER:-blender}"
  if ! "$BLENDER" --version >/dev/null 2>&1; then
    printf '%s\n' "Blender not found. Install Blender or set BLENDER=/path/to/blender."
    exit 1
  fi
  "$BLENDER" --background --python tools/make_collision_mesh.py -- assets/suzanne.glb assets/suzanne_collision.glb 96

shaders:
  shadercross shaders/hlsl/shader.hlsl -s HLSL -d SPIRV -t vertex -e VertexMain -o shaders/spv/shader.spv.vert
  shadercross shaders/hlsl/shader.hlsl -s HLSL -d SPIRV -t fragment -e FragmentMain -o shaders/spv/shader.spv.frag
  shadercross shaders/hlsl/shader.hlsl -s HLSL -d MSL -t vertex -e VertexMain -o shaders/msl/shader.msl.vert
  shadercross shaders/hlsl/shader.hlsl -s HLSL -d MSL -t fragment -e FragmentMain -o shaders/msl/shader.msl.frag
  shadercross shaders/hlsl/shader.hlsl -s HLSL -d DXIL -t vertex -e VertexMain -o shaders/dxil/shader.dxil.vert
  shadercross shaders/hlsl/shader.hlsl -s HLSL -d DXIL -t fragment -e FragmentMain -o shaders/dxil/shader.dxil.frag

check-shaders:
  #!/usr/bin/env sh
  before="$(shasum shaders/msl/shader.msl.vert shaders/msl/shader.msl.frag shaders/spv/shader.spv.vert shaders/spv/shader.spv.frag shaders/dxil/shader.dxil.vert shaders/dxil/shader.dxil.frag)"
  just shaders
  after="$(shasum shaders/msl/shader.msl.vert shaders/msl/shader.msl.frag shaders/spv/shader.spv.vert shaders/spv/shader.spv.frag shaders/dxil/shader.dxil.vert shaders/dxil/shader.dxil.frag)"
  test "$before" = "$after"

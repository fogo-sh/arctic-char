set windows-powershell

screenshot:
  cwebp screenshot.png -o screenshot.webp

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

build-and-run-map: build
  ./build/arctic-char Map

glsl-to-spv shader_name:
  glslc shaders/glsl/{{shader_name}}.glsl.vert -o shaders/spv/{{shader_name}}.spv.vert
  glslc shaders/glsl/{{shader_name}}.glsl.frag -o shaders/spv/{{shader_name}}.spv.frag

spv-to-msl shader_name:
  shadercross shaders/spv/{{shader_name}}.spv.vert -o shaders/msl/{{shader_name}}.msl.vert
  shadercross shaders/spv/{{shader_name}}.spv.frag -o shaders/msl/{{shader_name}}.msl.frag

spv-to-dxil shader_name:
  #!/usr/bin/env sh
  if [ "$(uname)" = "Windows_NT" ]; then
    shadercross shaders/spv/{{shader_name}}.spv.vert -o shaders/dxil/{{shader_name}}.dxil.vert
    shadercross shaders/spv/{{shader_name}}.spv.frag -o shaders/dxil/{{shader_name}}.dxil.frag
  else
    echo "Not rendering DXIL shaders"
  fi

shader shader_name: (glsl-to-spv shader_name) (spv-to-msl shader_name) (spv-to-dxil shader_name)

shaders: (shader "shader") (shader "ui")

map map_name:
  cd ./assets && qbsp -notex ./maps/{{map_name}}.map
  cd ./assets && vis ./maps/{{map_name}}.bsp
  cd ./assets && light ./maps/{{map_name}}.bsp

maps: (map "test")

build-atlas:
  odin run ./atlas-builder/

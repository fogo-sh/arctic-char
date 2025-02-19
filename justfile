set windows-powershell

build:
  odin build . -debug -out:arctic-char

run:
  ./arctic-char

build-and-run: build run

glsl-to-spv shader_name:
  glslc shaders/glsl/{{shader_name}}.glsl.vert -o shaders/spv/{{shader_name}}.spv.vert
  glslc shaders/glsl/{{shader_name}}.glsl.frag -o shaders/spv/{{shader_name}}.spv.frag

spv-to-msl shader_name:
  shadercross shaders/spv/{{shader_name}}.spv.vert -o shaders/msl/{{shader_name}}.msl.vert
  shadercross shaders/spv/{{shader_name}}.spv.frag -o shaders/msl/{{shader_name}}.msl.frag

spv-to-dxil shader_name:
  shadercross shaders/spv/{{shader_name}}.spv.vert -o shaders/dxil/{{shader_name}}.dxil.vert
  shadercross shaders/spv/{{shader_name}}.spv.frag -o shaders/dxil/{{shader_name}}.dxil.frag

shader shader_name: (glsl-to-spv shader_name) (spv-to-msl shader_name) (spv-to-dxil shader_name)

shaders: (shader "shader") (shader "text_shader")

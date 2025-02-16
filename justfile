build:
  odin build . -debug -out:arctic-char

run:
  ./arctic-char

build-and-run:
  just build
  just run

spv-to-msl:
  shadercross shaders/spv/shader.spv.vert -o shaders/msl/shader.msl.vert
  shadercross shaders/spv/shader.spv.frag -o shaders/msl/shader.msl.frag

glsl-to-spv:
  glslc shaders/glsl/shader.glsl.vert -o shaders/spv/shader.spv.vert
  glslc shaders/glsl/shader.glsl.frag -o shaders/spv/shader.spv.frag

shaders:
  just glsl-to-spv
  just spv-to-msl

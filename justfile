build:
  odin build . -debug -out:arctic-char

run:
  ./arctic-char

build-and-run:
  odin build . -debug -out:arctic-char && ./arctic-char

spv-to-msl:
  shadercross shaders/spv/shader.spv.vert -o shaders/msl/shader.msl.vert
  shadercross shaders/spv/shader.spv.frag -o shaders/msl/shader.msl.frag


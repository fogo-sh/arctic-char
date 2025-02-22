#!/usr/bin/env fish

set BUILD_DIR "./build"
set BINARY_NAME "arctic-char"

just build

mkdir -p $BUILD_DIR

# NOTE this sucks, we should be using ldd / etc. to find this, there must be a better way
# but also, this is just meant to be a build step, maybe we create a docker environment
# where this can be built?
set SDL_PATH "/usr/local/lib/libSDL3.so.0"
cp $SDL_PATH $BUILD_DIR/

patchelf --set-rpath '$ORIGIN' --force-rpath $BUILD_DIR/$BINARY_NAME

echo "Packaging complete! The application is in '$BUILD_DIR'."

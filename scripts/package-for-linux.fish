#!/usr/bin/env fish

set BUILD_DIR "./build"
set BINARY_NAME "arctic-char"

just build

mkdir -p $BUILD_DIR

set SDL_PATH (ldconfig -p | grep libSDL3.so.0 | awk '{print $NF}' | head -n 1)

if test -z "$SDL_PATH" -o ! -e "$SDL_PATH"
    echo "Error: libSDL3.so.0 not found!" >&2
    exit 1
end

echo "Found libSDL3 at: $SDL_PATH"

cp $BINARY_NAME $BUILD_DIR/
cp $SDL_PATH $BUILD_DIR/

patchelf --set-rpath '$ORIGIN' --force-rpath $BUILD_DIR/$BINARY_NAME

echo "Packaging complete! The application is in '$BUILD_DIR'."

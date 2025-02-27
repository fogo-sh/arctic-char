#!/bin/bash

docker build -t odin-appimage-builder .

docker run --rm \
    --privileged \
    -v $(pwd):/workspace:Z \
    odin-appimage-builder /bin/bash -c "
cd /workspace

echo 'Contents of /workspace:'
ls -l

echo 'Building Odin project...'
ranlib ./clay-odin/linux/clay.a
odin build . -out:arctic-char

if [ ! -f arctic-char ]; then
    echo 'Error: Binary arctic-char not found. Check your Odin project.'
    exit 1
fi

mkdir -p AppDir/usr/bin
cp arctic-char AppDir/usr/bin/

mkdir -p AppDir/usr/share/icons/hicolor/256x256/apps
cp arctic-char.png AppDir/usr/share/icons/hicolor/256x256/apps/arctic-char.png

mkdir -p AppDir/usr/share/applications
echo '[Desktop Entry]
Name=arctic-char
Icon=arctic-char
Exec=arctic-char
Type=Application
Categories=Utility;' > AppDir/usr/share/applications/arctic-char.desktop

echo 'Creating AppImage...'
linuxdeploy --appdir AppDir --output appimage

mv arctic-char*.AppImage arctic-char.AppImage
rm -rf AppDir
"

if [ -f arctic-char.AppImage ]; then
    echo "AppImage created: arctic-char.AppImage"
else
    echo "Error: AppImage creation failed."
fi

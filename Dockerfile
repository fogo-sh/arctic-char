FROM debian:trixie-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    clang \
    llvm-dev \
    libsdl3-dev \
    ca-certificates \
    fuse \
    libfuse2 \
    desktop-file-utils \
    libglib2.0-bin \
    patchelf \
    file \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage \
    && chmod +x appimagetool-x86_64.AppImage \
    && mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool

RUN wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage \
    && chmod +x linuxdeploy-x86_64.AppImage \
    && mv linuxdeploy-x86_64.AppImage /usr/local/bin/linuxdeploy

RUN git clone --depth 1 https://github.com/odin-lang/Odin.git /opt/odin \
    && cd /opt/odin \
    && make -C vendor/miniaudio/src \
    && make -C vendor/stb/src \
    && make -C vendor/cgltf/src \
    && make release \
    && ln -sf /opt/odin/odin /usr/local/bin/odin

ENV PATH="/opt/odin:${PATH}"

RUN clang --version \
    && odin version

CMD ["/bin/bash"]

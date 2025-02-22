# arctic char\*

![](./screenshot.webp|width=250)

as much as possible from Odin's core / vendor libaries are being used here:

- [SDL3](https://pkg.odin-lang.org/vendor/sdl3/): Windows, Audio, GPU, etc.
- [cgltf](https://pkg.odin-lang.org/vendor/cgltf/): gltf/glb handling
- [stb/image](https://pkg.odin-lang.org/vendor/stb/image/): reading images from disk into memory

however, we are pulling in:

- [clay](https://github.com/nicbarker/clay/tree/main/bindings/odin): UI (_being worked on as i write this version of the readme_)

using [SDL_shadercross](https://github.com/libsdl-org/SDL_shadercross) to compile shaders to different platforms.

---

todo:

- user interface
- main menu
- sounds on button press rather than just looping WAV
- billboard sprites

---

originally based on [nadako](https://github.com/nadako)'s [hello-sdlgpu3-odin](https://github.com/nadako/hello-sdlgpu3-odin), and also many aspects made following their amazing [youtube tutorial](https://www.youtube.com/playlist?list=PLI3kBEQ3yd-CbQfRchF70BPLF9G1HEzhy)!

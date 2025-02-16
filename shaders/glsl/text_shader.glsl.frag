#version 460

layout(set = 1, binding = 0) uniform TextColor {
    vec4 color;
} u_color;

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = u_color.color;
} 
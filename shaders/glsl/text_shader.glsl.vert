#version 460

layout(set = 1, binding = 0) uniform TextUBO {
    mat4 ortho;
} ubo;

layout(location = 0) in vec4 inPosition;

void main() {
    gl_Position = ubo.ortho * inPosition;
}

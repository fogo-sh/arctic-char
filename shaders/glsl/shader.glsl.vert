#version 460

layout(set=1, binding=0) uniform UBO {
    mat4 mvp;
    float time;
};

layout(location=0) in vec3 position;
layout(location=1) in vec4 color;
layout(location=2) in vec2 uv;

layout(location=0) out vec4 out_color;
layout(location=1) out vec2 out_uv;

float random(vec3 seed) {
    return fract(sin(dot(seed, vec3(12.9898, 78.233, 45.543))) * 43758.5453);
}

vec3 quantize(vec3 pos, float scale) {
    return floor(pos * scale) / scale;
}

void main() {
    float quantizationLevel = 16.0;
    float jitterAmount = 0.006;

    float stepSize = 0.5;
    float discreteTime = floor(time / stepSize) * stepSize;

    vec3 jitter = vec3(
        random(position + discreteTime) - 0.5,
        random(position.yzx + discreteTime) - 0.5,
        random(position.zxy + discreteTime) - 0.5
    ) * jitterAmount;

    vec3 quantizedPos = quantize(position + jitter, quantizationLevel);

    gl_Position = mvp * vec4(quantizedPos, 1.0);

    out_color = color;
    out_uv = uv;
}

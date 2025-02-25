#version 460

layout(set=1, binding=0) uniform UBO {
    mat4 mv;
    mat4 proj;
    vec2 viewport_size;
};

layout(location=0) in vec3 position;
layout(location=1) in vec4 color;
layout(location=2) in vec2 uv;

layout(location=0) out vec4 out_color;
layout(location=1) out vec2 out_uv;

vec3 quantize(vec3 pos, float scale) {
    float w = (proj * vec4(pos, 1.0)).w;
    return round(pos / w * scale) / scale * w;
}

void main() {
    vec3 viewPos = (mv * vec4(position, 1.0)).xyz;

    float jitter = 0.6;

    float z_orig = viewPos.z;
    float scale = (1.0 - jitter) * min(viewport_size.x, viewport_size.y) / 2.0;
    viewPos = quantize(viewPos, scale);

    viewPos.z = z_orig;

    gl_Position = proj * vec4(viewPos, 1.0);

    out_color = color;
    out_uv = uv;
}

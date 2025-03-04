#version 460

layout(set=1, binding=0) uniform UBO {
    float atlas_width;
    float atlas_height;
    vec4 atlas_lookup[4];
};

layout(location=0) in vec4 color;
layout(location=1) in vec3 uvw;

layout(location=0) out vec4 frag_color;

layout(set=2, binding=0) uniform sampler2D tex_sampler;

void main() {
    int texture_index = int(uvw.z);
    
    if (texture_index < 0 || texture_index > 3) {
        frag_color = vec4(1.0, 0.0, 0.0, 1.0); // Red for invalid index
        return;
    }
    
    vec4 texture_rect = atlas_lookup[texture_index];
    
    vec2 local_uv = fract(uvw.xy);
    
    float section_width = texture_rect.z / atlas_width;
    float section_height = texture_rect.w / atlas_height;
    
    float u = texture_rect.x / atlas_width + local_uv.x * section_width;
    float v = texture_rect.y / atlas_height + local_uv.y * section_height;
    
    vec2 atlas_uv = vec2(u, v);
    
    frag_color = texture(tex_sampler, atlas_uv) * color;
}

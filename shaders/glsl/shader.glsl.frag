#version 460

layout(location=0) in vec4 color;
layout(location=1) in vec3 uvw;

layout(location=0) out vec4 frag_color;

layout(set=2, binding=0) uniform sampler2D tex_sampler;

void main() {
	vec2 uv = uvw.xy;
	frag_color = texture(tex_sampler, uv) * color;
}

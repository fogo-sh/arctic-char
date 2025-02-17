#version 460

layout(location=0) out vec4 color;

void main() {
	vec2 uv = gl_FragCoord.xy / vec2(512, 512);
	
	color = vec4(uv.x, uv.y, 0.5, 1.0);
}
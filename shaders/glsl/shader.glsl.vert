#version 460

layout(location = 0) in vec4 inPosition;

layout(set=1, binding=0) uniform UBO {
	mat4 mvp;
};

void main() {
	gl_Position = mvp * vec4(inPosition[gl_VertexIndex]);
}
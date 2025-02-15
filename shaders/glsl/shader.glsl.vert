#version 460

layout(set=1, binding=0) uniform UBO {
	mat4 mvp;
};

layout(set = 0, location = 0) in vec4 inPosition;

void main() {
	gl_Position = mvp * inPosition[gl_VertexIndex];
}
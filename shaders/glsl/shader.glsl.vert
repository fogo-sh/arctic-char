#version 460

layout(set=1, binding=0) uniform UBO {
	mat4 mvp[128];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 vColor;

void main() {
	gl_Position = mvp[gl_InstanceIndex] * vec4(inPosition, 1.0);
	vColor      = inColor;
}

#version 460

layout(location = 0) in vec3 vertex;
layout(location = 1) in vec2 tex_coord;

//out vec2 UV;
out vec3 UVW;

uniform mat4 MVP;
uniform float stretchx;
uniform float stretchy;
uniform int depth;

// TODO
// get resolution for UVW calculation or precompute it and just send transformed depth to vertex shader

void main() {
	//UV = vec2(vertex.x / 2.f + 0.5f, vertex.y / 2.f + 0.5f);
	UVW = vec3(vertex.x / 2.f + 0.5f, vertex.y / 2.f + 0.5f, (1.f + 2.f * depth) / (2.f * 256.f));
	gl_Position = MVP * vec4(vec2(vertex.x * stretchx, vertex.y * stretchy), ((float(depth) / 255.f) * 2.f - 1.f) * stretchy, 1.0f);
}
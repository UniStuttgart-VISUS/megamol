#version 430

layout(location = 0) in vec2 pos;
layout(location = 1) in float radiusIn;
layout(location = 2) in float typeIn;

layout(location = 0) out float radius;
layout(location = 1) out float type;

void main() {
    radius = radiusIn;
    type = typeIn;
    gl_Position = vec4(pos, 0.0, 1.0);
}

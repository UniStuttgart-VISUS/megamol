#version 430

layout(location = 0) in vec2 pos;
layout(location = 1) in float radiusIn;

out float radius;

void main() {
    radius = radiusIn;
    gl_Position = vec4(pos, 0.0, 1.0);
}

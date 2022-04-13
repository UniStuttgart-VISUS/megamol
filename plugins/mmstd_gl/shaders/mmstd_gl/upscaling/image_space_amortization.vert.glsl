#version 450

#include "../common/quad_vertices.inc.glsl"

out vec2 uvCoords;

void main() {
    vec2 coord = quadVertexPosition();

    gl_Position = vec4(2.0f * coord - 1.0f, 0.0f, 1.0f);
    uvCoords = coord;
}

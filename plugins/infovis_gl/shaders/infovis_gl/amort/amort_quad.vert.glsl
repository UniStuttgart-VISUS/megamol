#version 450

out vec2 uvCoords;

void main() {
    const vec2 coords[4] = vec2[4](
        vec2(0.0f, 0.0f),
        vec2(1.0f, 0.0f),
        vec2(0.0f, 1.0f),
        vec2(1.0f, 1.0f));

    const vec2 coord = coords[gl_VertexID];

    gl_Position =  vec4(2.0f * coord - 1.0f, 0.0f, 1.0f);
    uvCoords = coord;
}

#version 430

out vec2 uvCoords;

void main() {
    const vec4 vertices[6] = vec4[6](
        vec4(-1.0, -1.0, 0.0, 0.0),
        vec4(1.0, 1.0, 1.0, 1.0),
        vec4(-1.0, 1.0, 0.0, 1.0),
        vec4(1.0, 1.0, 1.0, 1.0),
        vec4(-1.0, -1.0, 0.0, 0.0),
        vec4(1.0, -1.0, 1.0, 0.0));

    vec4 vertex = vertices[gl_VertexID];

    gl_Position =  vec4(vertex.xy, -1.0, 1.0);
    uvCoords = vertex.zw;
}

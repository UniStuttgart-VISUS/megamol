#version 430

out vec2 vsUV;

void main() {
    const vec2 quad[4] = { vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0) };
    vsUV = quad[gl_VertexID] * 0.5 + 0.5;
    gl_Position = vec4(quad[gl_VertexID], 0.0, 1.0);
}

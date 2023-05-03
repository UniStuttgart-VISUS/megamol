#version 430

layout(location = 0) in vec3 in_pos;
uniform mat4 mvp;
uniform int line_len;

out vec4 vertColor;

void main(void) {
    vertColor = vec4(gl_VertexID / float(line_len) * 0.5 + 0.5, 0.5, 0.5, 1.0);
    gl_Position = mvp * vec4(in_pos, 1.0);
}

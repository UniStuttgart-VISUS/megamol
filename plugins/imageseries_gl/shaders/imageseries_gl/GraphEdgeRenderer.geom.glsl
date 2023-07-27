#version 430

uniform mat4 matrix;
uniform float width;

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in float weight[2];

layout(location = 0) out float edgeWeight;

void main() {
    vec2 from = gl_in[0].gl_Position.xy;
    vec2 to = gl_in[1].gl_Position.xy;

    vec2 direction = to - from;
    vec2 orth = vec2(-direction.y, direction.x);
    orth = orth / sqrt(orth.x * orth.x + orth.y * orth.y);

    edgeWeight = weight[0];
    gl_Position = matrix * vec4(from + width * orth, 0.0, 1.0);
    EmitVertex();

    edgeWeight = weight[1];
    gl_Position = matrix * vec4(to, 0.0, 1.0);
    EmitVertex();

    edgeWeight = weight[0];
    gl_Position = matrix * vec4(from, 0.0, 1.0);
    EmitVertex();

    edgeWeight = weight[0];
    gl_Position = matrix * vec4(from - width * orth, 0.0, 1.0);
    EmitVertex();

    EndPrimitive();
}

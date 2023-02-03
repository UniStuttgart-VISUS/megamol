#version 430

uniform mat4 matrix;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in float radius[];

out vec2 texCoord;

void main() {
    vec2 center = gl_in[0].gl_Position.xy;

    gl_Position = matrix * vec4(center + vec2(-radius[0], -radius[0]), 0.0, 1.0);
    texCoord = vec2(-1.0, -1.0);
    EmitVertex();

    gl_Position = matrix * vec4(center + vec2(radius[0], -radius[0]), 0.0, 1.0);
    texCoord = vec2(1.0, -1.0);
    EmitVertex();

    gl_Position = matrix * vec4(center + vec2(-radius[0], radius[0]), 0.0, 1.0);
    texCoord = vec2(-1.0, 1.0);
    EmitVertex();

    gl_Position = matrix * vec4(center + vec2(radius[0], radius[0]), 0.0, 1.0);
    texCoord = vec2(1.0, 1.0);
    EmitVertex();

    EndPrimitive();
}

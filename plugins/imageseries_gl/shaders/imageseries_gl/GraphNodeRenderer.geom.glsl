#version 430

uniform mat4 matrix;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in float radius[1];
layout(location = 1) in float type[1];

layout(location = 0) out vec2 texCoord;
layout(location = 1) out float nodeType;

void main() {
    const vec2 center = gl_in[0].gl_Position.xy;
    const float eff_radius = radius[0];

    gl_Position = matrix * vec4(center + vec2(-eff_radius, -eff_radius), 0.0, 1.0);
    texCoord = vec2(-1.0, -1.0);
    nodeType = type[0];
    EmitVertex();

    gl_Position = matrix * vec4(center + vec2(eff_radius, -eff_radius), 0.0, 1.0);
    texCoord = vec2(1.0, -1.0);
    nodeType = type[0];
    EmitVertex();

    gl_Position = matrix * vec4(center + vec2(-eff_radius, eff_radius), 0.0, 1.0);
    texCoord = vec2(-1.0, 1.0);
    nodeType = type[0];
    EmitVertex();

    gl_Position = matrix * vec4(center + vec2(eff_radius, eff_radius), 0.0, 1.0);
    texCoord = vec2(1.0, 1.0);
    nodeType = type[0];
    EmitVertex();

    EndPrimitive();
}

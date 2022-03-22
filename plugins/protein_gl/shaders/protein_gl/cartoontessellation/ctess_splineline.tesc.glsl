#version 430

uniform int uOuter0 = 1;
uniform int uOuter1 = 16;

layout(vertices = 4) out;

layout(packed, binding = 2) buffer shader_data {
    vec4 positionsCa[];
};

void main() {
    gl_out[gl_InvocationID].gl_Position = positionsCa[gl_PrimitiveID + gl_InvocationID];
    gl_TessLevelOuter[0] = float(uOuter0);
    gl_TessLevelOuter[1] = float(uOuter1);
}

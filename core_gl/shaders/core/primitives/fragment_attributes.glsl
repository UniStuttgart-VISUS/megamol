#extension GL_ARB_explicit_attrib_location : require // glsl version 130

flat in vec4 color;
flat in vec2 center;
flat in float radius;
in vec2 texcoord;
flat in vec4 attributes;

uniform vec2 viewport;
uniform sampler2D tex;
uniform int apply_smooth;

layout(location = 0) out vec4 outColor;

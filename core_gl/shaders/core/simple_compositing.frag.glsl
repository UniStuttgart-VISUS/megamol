#version 130

#extension GL_ARB_explicit_attrib_location : require // glsl version 130

uniform sampler2D color_tex;
uniform sampler2D depth_tex;

uniform vec2 viewport;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uvcoord = vec2(gl_FragCoord.x / viewport.x, gl_FragCoord.y / viewport.y);
    vec4 color = texture2D(color_tex, uvcoord);
    float depth = texture2D(depth_tex, uvcoord).x;
    gl_FragDepth = depth;
    outColor = color;
}

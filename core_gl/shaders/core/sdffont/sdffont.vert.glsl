#version 400

#ifdef GLOBAL_COLOR
uniform vec4 inColor;
#else
layout (location = 2) in vec4 inColor;
#endif

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inTexCoord;

uniform float fontSize;
uniform mat4 mvpMat;

out vec2 texCoord;
out vec4 color;

void main() {
    texCoord  = inTexCoord;
    color = inColor;
    gl_Position = mvpMat * vec4(inPos, 1.0);
}

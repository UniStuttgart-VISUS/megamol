uniform mat4 MVP;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 tex;

out vec2 texCoord;

void main(void) {
    texCoord = tex;
    gl_Position = MVP * vec4(pos, 0.0, 1.0);
}
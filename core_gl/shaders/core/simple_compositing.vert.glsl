#version 130

uniform mat4 mvp;
uniform vec2 viewport;

void main(void) {
float right = viewport.x;
float left = 0.0;
float up = viewport.y;
float bottom = 0.0;

vec3 pos[4] =
    vec3[](vec3(left, bottom, 0.0), vec3(right, bottom, 0.0), vec3(right, up, 0.0), vec3(left, up, 0.0));

    vec4 inPos = mvp * vec4(pos[gl_VertexID], 1.0);
    inPos /= inPos.w;
    gl_Position = inPos;
}

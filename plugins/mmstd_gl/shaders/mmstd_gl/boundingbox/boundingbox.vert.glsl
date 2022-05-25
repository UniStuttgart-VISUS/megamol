#version 330

layout(location = 0) in vec3 in_position;

uniform vec3 bbMin;
uniform vec3 bbMax;
uniform mat4 mvp;

void main(void) {
    float x = (in_position.x < 0.0) ? bbMin.x : bbMax.x;
    float y = (in_position.y < 0.0) ? bbMin.y : bbMax.y;
    float z = (in_position.z < 0.0) ? bbMin.z : bbMax.z;
    gl_Position = mvp * vec4(x, y, z, 1.0);
}

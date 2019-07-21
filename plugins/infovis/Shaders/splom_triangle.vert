uniform mat4 modelViewProjection;

in vec2 position;
in float value;

out vec3 vsPoint;

void main() {
    vsPoint = vec3(position, value);
    gl_Position = modelViewProjection * vec4(position, 0.0, 1.0);
}

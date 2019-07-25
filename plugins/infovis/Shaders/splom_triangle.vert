uniform mat4 modelViewProjection;

in vec2 position;
in float value;

out float vsValue;

void main() {
    vsValue = value;
    gl_Position = modelViewProjection * vec4(position, 0.0, 1.0);
}

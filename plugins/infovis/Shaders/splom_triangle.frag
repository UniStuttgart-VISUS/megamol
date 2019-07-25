in float vsValue;

layout(location = 0) out vec4 fsColor;

void main() {
    fsColor = vec4(vsValue, vsValue, vsValue, 1.0);
}

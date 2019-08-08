in float vsValue;
in vec4 vsValueColor;

layout(location = 0) out vec4 fsColor;

void main() {
    fsColor = toScreen(vsValue, vsValueColor, 1.0);
}

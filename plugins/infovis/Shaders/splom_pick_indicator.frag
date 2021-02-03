uniform vec4 indicatorColor = vec4(0.0, 0.0, 1.0, 1.0);

in vec2 circleCoord;

out vec4 fragColor;

void main()
{
    float dist = length(circleCoord);

    if (dist < 1.0) {
        fragColor = indicatorColor;
    } else {
        discard;
    }
}

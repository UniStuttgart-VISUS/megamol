uniform uint depth;

in vec4 vsColor;
in vec2 vsFragCoord; // in object space
in vec2 vsSmallSteppSize;

out vec4 fsColor;

float rayCastSmall(vec2 position) {
    float width = 10.0;
    float len = 0.2;

    vec2 frac = position / vsSmallSteppSize;
    vec2 integer = roundEven(frac);

    if (abs(frac - integer).x > len || abs(frac - integer).y > len) return 0.0;

    return mix(1.0, 0.0, min(abs(frac - integer).x * width, 1.0)) +
           mix(1.0, 0.0, min(abs(frac - integer).y * width, 1.0));
}

float rayCastMid(vec2 position) {
    float width = 14.0;
    float len = 0.2;

    vec2 frac = position / (vsSmallSteppSize * 5);
    vec2 integer = roundEven(frac);

    if (abs(frac - integer).x > len || abs(frac - integer).y > len) return 0.0;

    return mix(1.0, 0.0, min(abs(frac - integer).x * width, 1.0)) +
           mix(1.0, 0.0, min(abs(frac - integer).y * width, 1.0));
}

float rayCastBig(vec2 position) {
    float width = 18.0;
    float len = 0.2;

    vec2 frac = position / (vsSmallSteppSize * 5 * 5);
    vec2 integer = roundEven(frac);

    if (abs(frac - integer).x > len || abs(frac - integer).y > len) return 0.0;

    return mix(1.0, 0.0, min(abs(frac - integer).x * width, 1.0)) +
           mix(1.0, 0.0, min(abs(frac - integer).y * width, 1.0));
}


float rayCast(vec2 position) {
    float value = 0;
    if (depth >= 1) {
        value += rayCastBig(position);
    }
    if (depth >= 2) {
        value += rayCastMid(position);
    }
    if (depth >= 3) {
        value += rayCastSmall(position);
    }

    return min(1.0, value);
}

void main(void) {
    float d = rayCast(vsFragCoord);
    if (d == 0.0) {
        discard;
    }
    fsColor = vec4(vsColor.rgb, d);
}

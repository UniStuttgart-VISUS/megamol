#version 430

uniform uint depth;

in vec4 vsColor;
in vec2 vsFragCoord; // in object space
in vec2 vsSmallSteppSize;

out vec4 fsColor;

float roundedBox(vec2 position, vec2 size, float radius) {
    return length(max(abs(position) - size, 0.0)) - radius;
}

float marker(vec2 position, float stepFactor) {
    vec2 frac = position / (vsSmallSteppSize * stepFactor);
    vec2 integer = roundEven(frac);

    float stepLen = 0.2;
    float lineWidth = 0.0025;
    float radius = 0.01;
    float hBox = roundedBox(frac - integer, vec2(stepLen, lineWidth), radius);
    float vBox = roundedBox(frac - integer, vec2(lineWidth, stepLen), radius);

    return min(vBox, hBox) * stepFactor;
}

float rayCast(vec2 position) {
    float d = 1.0;
    if (depth >= 1) {
        d = min(d, marker(position, 5.0 * 5.0));
    }
    if (depth >= 2) {
        d = min(d, marker(position, 5.0));
    }
    if (depth >= 3) {
        d = min(d, marker(position, 1.0));
    }
    return smoothstep(0.5 / pow(float(depth), 2), 0.0, d);
}

void main(void) {
    float rho = rayCast(vsFragCoord);
    if (rho == 0.0) {
        discard;
    }
    fsColor = vec4(vsColor.rgb, vsColor.a * rho);
}

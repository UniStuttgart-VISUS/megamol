#version 430

#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"
#include "mmstd_gl/flags/bitflags.inc.glsl"
#include "common/mapping.inc.glsl"

uniform int kernelType;

in float gsValue;
in vec4 gsValueColor;
in vec2 gsLineSize;
in vec2 gsLineCoord;

out vec4 fsColor;

float erf(float x) {
    const float PI = 3.1415926535897932384626433832795;
    return 2.0 / sqrt(PI) * sign(x) * sqrt(1.0 - exp(-pow(x, 2.0))) * (sqrt(PI) / 2.0 + 31.0 / 200.0 * exp(-pow(x, 2.0)) - 341.0 / 8000.0 * exp(-2.0 * pow(x, 2.0)));
}

// http://www.wolframalpha.com/input/?i=integrate+e%5E(-0.5*(h%5E2%2Bx%5E2))%2F(sigma+*+2.506628274631000502415765284811)+dx+from+x%3D0+to+l%2F2
float integralGauss2(float h, float l) {
    float sigma = 2.0;
    return 0.5 * exp(-0.5 * h * h) * erf(0.353553 * l) / sigma;
}

float movingKernel(vec2 position, float width, float len) {
    // Intersection between circle and center line.
    float delta = pow(width, 2.0) - pow(position.y, 2.0);
    if (delta <= 0) {
        // Less than two intersections.
        return 0;
    }
    delta = sqrt(delta);

    // Compute two intersected center points.
    float x1 = position.x + delta;
    float x2 = position.x - delta;
    if (x1 < 0) {
        x1 = 0;
    } else if (x1 > len) {
        x1 = len;
    }
    if (x2 < 0) {
        x2 = 0;
    } else if (x2 > len) {
        x2 = len;
    }

    // Compute the distance between the two center points.
    float density;
    switch (kernelType) {
    case 0:
        density = abs(x1 - x2) / (2.0 * width);
        break;
    case 1:
        // Integrate gauss2(x*6) dx from 0 to half distance with normalized distance and position.
        density = 2.0 * integralGauss2(3.0 * position.y / width, 3.0 * abs(x1 - x2) / (2.0 * width));
        break;
    default:
        density = 0.0;
    }
    return density;
}

void main(void) {
    float density = movingKernel(gsLineCoord, gsLineSize.x, gsLineSize.y);

    fsColor = toScreen(gsValue, gsValueColor, density);
}

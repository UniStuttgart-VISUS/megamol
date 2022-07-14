#version 430

#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"
#include "mmstd_gl/flags/bitflags.inc.glsl"
#include "common/mapping.inc.glsl"

uniform int kernelType;

in float vsKernelSize;
in float vsPixelKernelSize;
in float vsValue;
in vec4 vsValueColor;

out vec4 fsColor;

// 2D circular kernel.
float circle2(vec2 p) {
    return length(p) < 0.5 ? 1.0 : 0.0;
}

// 2D gaussian kernel, leveled to zero and scaled to be [0.0;1.0].
float gauss2(vec2 p) {
    const vec2 x = p * 6.0;
    const float g = exp(-dot(x, x) * 0.5);
    const float gAtPlusMinus3 = exp(-4.5);
    return max((g - gAtPlusMinus3) / (1.0 - gAtPlusMinus3), 0.0f);
}

void main(void) {
    const vec2 distance = gl_PointCoord.xy - vec2(0.5);
    float density;
    switch (kernelType) {
    case 0:
        density = circle2(distance);
        break;
    case 1:
        density = gauss2(distance);
        break;
    default:
        density = 0.0;
    }

    if (density <= 0.0f) {
        discard;
    }

    const float attenuation = vsPixelKernelSize - vsKernelSize;
    density *= pow(1.0 - attenuation, 2);

    fsColor = toScreen(vsValue, vsValueColor, density);
}

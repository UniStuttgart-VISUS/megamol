#version 430

#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
#include "core/bitflags.inc.glsl"
#include "splom_common/splom_mapping.inc.glsl"
#include "splom_common/splom_plots.inc.glsl"
#include "splom_common/splom_data.inc.glsl"

uniform ivec2 viewRes;
uniform mat4 modelViewProjection;

uniform int rowStride;

uniform float kernelWidth;
uniform int kernelType;

in float vsKernelSize;
in vec2 vsPosition;
flat in vec2 vsPointPosition;
flat in int vsPointID;

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

    uint flag = flags[vsPointID];
    if(!(bitflag_test(flag, FLAG_ENABLED | FLAG_FILTERED, FLAG_ENABLED))) {
        discard;
    }

    const vec2 distance = ((vsPosition.xy - vsPointPosition.xy) / vsKernelSize) * 0.5;

    if (length(distance) > 0.5f) {
        discard;
    }

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

    const int rowOffset = vsPointID * rowStride;
    float value = 1.0;

    if (valueColumn == -1) {
        value = 1.0;
    } else {
        value = values[rowOffset + valueColumn];
    }
    vec4 valueColor = flagifyColor(tflookup(value), flag);

    // compute attenuation factor in screen space
    const float attenuation = 1.0 - kernelWidth/vsKernelSize;
    density *= pow(1.0 - attenuation, 2);

    fsColor = toScreen(value, valueColor, density);
}

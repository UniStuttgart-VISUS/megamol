#version 430

#include "mmstd_gl/common/tfconvenience.inc.glsl"

uniform sampler2D image;
uniform int displayMode;

in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

vec4 labelColor(const float value) {
    const vec4 table[] = vec4[18](
        vec4(0.9, 0.1, 0.1, 1.0),
        vec4(0.1, 0.9, 0.1, 1.0),
        vec4(0.9, 0.9, 0.1, 1.0),
        vec4(0.1, 0.1, 0.9, 1.0),
        vec4(0.9, 0.1, 0.9, 1.0),
        vec4(0.1, 0.9, 0.9, 1.0),

        vec4(0.7, 0.1, 0.1, 1.0),
        vec4(0.1, 0.7, 0.1, 1.0),
        vec4(0.7, 0.7, 0.1, 1.0),
        vec4(0.1, 0.1, 0.7, 1.0),
        vec4(0.7, 0.1, 0.7, 1.0),
        vec4(0.1, 0.7, 0.7, 1.0),

        vec4(0.9, 0.3, 0.3, 1.0),
        vec4(0.3, 0.9, 0.3, 1.0),
        vec4(0.9, 0.9, 0.3, 1.0),
        vec4(0.3, 0.3, 0.9, 1.0),
        vec4(0.9, 0.3, 0.9, 1.0),
        vec4(0.3, 0.9, 0.9, 1.0));

    return table[int(floor(value)) % 18];
}

vec4 getColor(vec4 color) {
    float value = color.r;

    switch (displayMode) {
    default:
        return color;
    case 1:
    case 3:
        value *= pow(2.0, 8.0) - 1;

        if (value < 2.0) {
            return vec4(0.0, 0.0, 0.0, 1.0);
        }

        break;
    case 2:
    case 4:
        value *= pow(2.0, 16.0) - 1;

        if (value < 1.0) {
            return vec4(0.0, 0.0, 0.0, 1.0);
        } else if (value > 65532.0) {
            return vec4(1.0, 1.0, 1.0, 1.0);
        }
    }

    switch (displayMode) {
    case 1:
    case 2:
        return tflookup(value);
    case 3:
    case 4:
        return labelColor(value);
    }
}

void main() {
    fragColor = getColor(texture(image, texCoord));
}

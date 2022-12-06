#version 130

#include "core/primitives/fragment_attributes.glsl"
#include "core/primitives/smoothing.glsl"

void main() {
    float alpha = 1.0;
    float distance = length(center - gl_FragCoord.xy);
    if (bool(apply_smooth)) {
        distance = 1.0 - (distance / (radius * 2.0));
        alpha = smoothing(distance);
    }
    else {
        if (distance > radius) {
            discard;
        }
    }
    outColor = vec4(color.rgb, color.a * alpha);
}

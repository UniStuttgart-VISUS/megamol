#version 130

#include "core/primitives/fragment_attributes.glsl"
#include "core/primitives/smoothing.glsl"

void main() {
    vec4 texcolor =  texture(tex, texcoord);
    float forced_alpha = attributes.x;
    float alpha = texcolor.a;
    if (forced_alpha > 0.0) {
        alpha = forced_alpha;
    }
    if (alpha <= 0.0) {
        discard;
    }
    outColor = vec4(texcolor.rgb, alpha);

    if ((forced_alpha <= 0.0) && (color.a > 0.0)) {
        outColor = vec4(color.rgb, alpha);
    }
}

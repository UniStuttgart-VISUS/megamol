#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"

layout(binding = 1) uniform sampler2D fragmentCountTex;

smooth in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

void main() {
    float fragmentCount = texture(fragmentCountTex, texCoord).r;
    if (fragmentCount > 0) {
        fragColor = tflookup(fragmentCount);
        //fragColor = vec4(0.5);}
    }else{
        discard;
    }
}

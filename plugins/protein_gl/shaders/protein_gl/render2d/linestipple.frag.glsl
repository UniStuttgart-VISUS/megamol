#version 460

layout(location = 0) out vec4 color_out;

uniform uint pattern = 0x5555;
uniform float factor = 1.0;
uniform vec2 viewport;

flat in vec3 startPos;
in vec3 pos;
in vec3 color;

void main(void) {
    vec2 dir = (pos.xy - startPos.xy) * viewport / 2.0;
    float dist = length(dir);

    uint bit = uint(round(dist / factor)) & 15U;
    if((pattern & (1U << bit)) == 0U) {
        discard;
    }

    color_out = vec4(color, 1);
}
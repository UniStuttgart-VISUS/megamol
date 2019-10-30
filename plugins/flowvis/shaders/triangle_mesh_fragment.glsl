in vec4 color;
in vec3 normal;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out float frag_depth;

void main(void) {
    frag_color = color.rgb;
    frag_normal = normal;
    frag_depth = gl_FragCoord.z;
}

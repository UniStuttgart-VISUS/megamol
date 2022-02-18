in vec3 normal;
in vec2 uv;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;

void main(void) {
    albedo_out = vec4(0.6,0.6,0.6,1.0);
    normal_out = normalize(normal);
    depth_out = gl_FragCoord.z;
}

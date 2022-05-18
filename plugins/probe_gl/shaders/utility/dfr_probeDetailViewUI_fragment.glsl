layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;

void main(void) {
    albedo_out = vec4(1.0,1.0,1.0,1.0);
    normal_out = vec3(1.0);
    depth_out = gl_FragCoord.z;
}

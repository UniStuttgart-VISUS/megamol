layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 world_pos;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;

void main(void) {
    albedo_out = vec4(1.0,0.0,1.0,1.0);
    //normal_out = normal;
    normal_out = normalize(cross(dFdxFine(world_pos),dFdyFine(world_pos)));
    depth_out = gl_FragCoord.z;
}
layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 debug_col;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;

void main(void) {
    //albedo_out = vec4(world_pos/vec3(50.0,50.0,-170.0),1.0);
    //albedo_out = vec4(0.57,0.05,0.05,1.0);
    albedo_out = vec4(0.7,0.7,0.7,1.0);
    normal_out = normal;
    depth_out = gl_FragCoord.z;
}
in vec3 normal;
in vec3 world_pos;

out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main(void) {
    albedo_out = vec3(world_pos/vec3(50.0,50.0,-170.0,1.0));
    normal_out = normal;
    depth_out = gl_FragCoord.z;
}
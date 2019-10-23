out layout(location = 0) vec3 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main(void) {
    albedo_out = vec3(1.0);
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;
}
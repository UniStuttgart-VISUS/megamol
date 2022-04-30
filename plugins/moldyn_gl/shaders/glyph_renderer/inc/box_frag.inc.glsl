in vec4 ws_pos;
in vec4 vert_color;

in vec3 inv_rad;

in flat vec3 dir_color;

in flat vec3 normal;
in flat vec3 transformed_normal;
in vec3 view_ray;

//layout (location = 0) out vec4 out_frag_color;
out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    depth_out = gl_FragCoord.z;
    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation),1.0);
    normal_out = transformed_normal;
}

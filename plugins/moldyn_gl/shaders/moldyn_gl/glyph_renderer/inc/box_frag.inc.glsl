in vec4 vert_color;
in flat vec3 dir_color;
in flat vec3 tensor_space_normal;

out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation),1.0);
    normal_out = tensor_space_normal;

    // can be removed when depth attachment will be removed
    // (not used in SimpleRenderTarget)
    depth_out = gl_FragCoord.z;
}

#extension GL_ARB_bindless_texture : require

struct MeshShaderParams
{
    vec4 glpyh_position;
    uvec2 texture_handle;
    float slice_idx;
    float padding0;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

layout(location = 0) flat in int draw_id;
layout(location = 1) in vec2 uv_coords;

out layout(location = 0) vec3 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    sampler2DArray tx_hndl = sampler2DArray(mesh_shader_params[draw_id].texture_handle);
    vec3 tx_crds = vec3(uv_coords,mesh_shader_params[draw_id].slice_idx);

    vec3 tx_col = texture( tx_hndl,tx_crds).rgb;
    albedo_out = tx_col;
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;
}
#extension GL_ARB_shader_draw_parameters : require
#extension GL_ARB_bindless_texture : require

struct MeshShaderParams
{
    vec4 glpyh_position;
    sampler2D texture_handle;
    float padding0;
    float padding1;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

flat in layout(location = 0) int draw_id;
in layout(location = 1) vec2 uv_coords;

out layout(location = 0) vec3 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main(void) {
    //albedo_out = vec3(1.0);
    vec3 tx_col = texture(mesh_shader_params[draw_id].texture_handle,uv_coords).rgb;
    albedo_out = tx_col;
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;
}
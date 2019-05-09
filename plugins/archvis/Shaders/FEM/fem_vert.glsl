#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams
{
    mat4 transform;
};

struct MaterialShaderParams
{
    uvec2 texture_handle_dummy;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };
layout(std430, binding = 1) readonly buffer MaterialShaderParamsBuffer { MaterialShaderParams[] mtl_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec3 v_position;

out vec3 vColour;

void main()
{
    mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;
    vColour = (object_transform * vec4(v_position,1.0)).xyz;
    gl_Position =  proj_mx * view_mx * object_transform * vec4(v_position,1.0);
}
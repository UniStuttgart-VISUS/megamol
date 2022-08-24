#version 450

#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams
{
    mat4 transform;
    float force;
    float padding0;
    float padding1;
    float padding2;
};

struct MaterialShaderParams
{
    uvec2 texture_handle_dummy;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };
layout(std430, binding = 1) readonly buffer MaterialShaderParamsBuffer { MaterialShaderParams[] mtl_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec3 v_normal;
in vec3 v_position;
//in vec4 v_tangent;
//in vec2 v_uv;

out vec3 world_pos;
out vec3 normal;
out float force;

void main()
{
    //gl_Position = vec4(v_position.xy, 0.5 ,1.0);
    mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;
    world_pos = (object_transform * vec4(v_position,1.0)).xyz;
    normal = inverse(transpose(mat3(object_transform))) * v_normal;
    force = mesh_shader_params[gl_DrawIDARB].force;
    gl_Position =  proj_mx * view_mx * object_transform * vec4(v_position,1.0);

}

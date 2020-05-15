#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams
{
    mat4 transform;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

//layout(location = 0) in vec3 v_normal;
layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
//layout(location = 2) in vec4 v_tangent;
//layout(location = 3) in vec2 v_uv;

layout(location = 0) out vec3 world_pos;
layout(location = 1) out vec3 normal;

void main()
{
    normal = inverse(transpose(mat3(mesh_shader_params[gl_DrawIDARB].transform))) * v_normal;
    world_pos = v_position;
    
    mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;
    //gl_Position =  proj_mx * view_mx * object_transform * vec4(v_position,1.0);
    //gl_Position =  object_transform * vec4(v_position,1.0);
    gl_Position =  vec4(v_position,1.0);
}
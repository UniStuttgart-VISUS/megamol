#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams
{
    mat4 transform;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams mesh_shader_params[]; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_position;
layout(location = 2) in vec4 v_tangent;
layout(location = 3) in vec2 v_uv;

out vec3 vWorldPos;
out vec3 vNormal;

void main()
{
    //gl_Position = vec4(v_position.xy, 0.5 ,1.0);
    vNormal = v_normal;
    mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;
    vWorldPos = (object_transform * vec4(v_position,1.0)).xyz;
    gl_Position =  proj_mx * view_mx * object_transform * vec4(v_position,1.0);

}
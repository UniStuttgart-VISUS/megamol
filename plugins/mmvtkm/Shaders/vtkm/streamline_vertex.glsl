#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams
{
    mat4 transform;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_color;

out vec3 color;

void main()
{
    //gl_Position = vec4(v_position.xy, 0.5 ,1.0);
    gl_Position =  proj_mx * view_mx * vec4(v_position,1.0);
	color = v_color;
}
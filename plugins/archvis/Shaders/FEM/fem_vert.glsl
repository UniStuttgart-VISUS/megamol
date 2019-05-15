#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams
{
    mat4 transform;
};

struct NodeDeformation
{
    vec4 deformation;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };
layout(std430, binding = 1) readonly buffer NodeDeformationsBuffer { NodeDeformation[] node_deformations; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec3 v_position;

out vec3 vColour;

void main()
{
    mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;
    vec3 vertex_displacement = (node_deformations[gl_VertexID].deformation).xyz;
    vColour = abs(vertex_displacement) * 10.0f;
    gl_Position =  proj_mx * view_mx * object_transform * vec4(v_position + vertex_displacement,1.0);
}
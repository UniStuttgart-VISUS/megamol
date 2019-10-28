#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams
{
    mat4 transform;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams mesh_shader_params[]; };

layout(location = 0) in vec3 position;
layout(location = 1) in float value;

out vec4 colors;

void main() {
    colors = vec4(value);

    const mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;
    gl_Position =  object_transform * vec4(position, 1.0f);
}

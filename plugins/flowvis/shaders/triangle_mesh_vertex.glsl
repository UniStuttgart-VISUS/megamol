#extension GL_ARB_shader_draw_parameters : require
#extension GL_ARB_bindless_texture : require

struct MeshShaderParams
{
    sampler1D transfer_function;
    float min_value;
    float max_value;
    int use_clip_plane;
    vec4 clip_plane;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams mesh_shader_params[]; };

layout(location = 0) in vec3 position;
layout(location = 1) in float value;

out vec4 colors;

void main() {
    colors = texture(mesh_shader_params[gl_DrawIDARB].transfer_function,
        (mesh_shader_params[gl_DrawIDARB].min_value == mesh_shader_params[gl_DrawIDARB].max_value)
            ? 0.5f
            : ((value - mesh_shader_params[gl_DrawIDARB].min_value) /
                  (mesh_shader_params[gl_DrawIDARB].max_value - mesh_shader_params[gl_DrawIDARB].min_value)));

    if (mesh_shader_params[gl_DrawIDARB].use_clip_plane != 0 && clip_halfspace(position,
        mesh_shader_params[gl_DrawIDARB].clip_plane)) {

        colors.a = 0.0f;
    }

    gl_Position =  vec4(position, 1.0f);
}

#version 450
#extension GL_ARB_bindless_texture : require

struct MeshShaderParams
{
    vec4 glpyh_position;
    vec4 probe_direction;
    float scale;

    int probe_id;
    int state;

    float sample_cnt;
    vec4 samples[32];  

    uvec2 tf_texture_handle;
    float tf_min;
    float tf_max;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 world_pos;
layout(location = 2) in float sample_value;
layout(location = 3) flat in int draw_id;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;

void main(void) {
    sampler2D tf_tx = sampler2D(mesh_shader_params[draw_id].tf_texture_handle);
    float tf_min = mesh_shader_params[draw_id].tf_min;
    float tf_max = mesh_shader_params[draw_id].tf_max;
    vec3 colour = texture(tf_tx, vec2((sample_value - tf_min) / (tf_max-tf_min), 0.5) ).rgb;

    //albedo_out = vec4(1.0,0.0,1.0,1.0);
    albedo_out = vec4(colour,1.0);
    //normal_out = normal;
    normal_out = normalize(cross(dFdxFine(world_pos),dFdyFine(world_pos)));
    depth_out = gl_FragCoord.z;
}

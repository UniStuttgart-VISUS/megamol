#version 450
#extension GL_ARB_bindless_texture : require


struct MeshShaderParams {
    mat4 transform;
    uvec2 albedo_texture_handle;
    uvec2 normal_texture_handle;
    uvec2 metallicRoughness_texture_handle;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams mesh_shader_params[]; };

struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer LightParamsBuffer { LightParams light_params[]; };

in vec3 vnormal;
in vec2 uv_coord;
in mat3 tangent_space_matrix;
flat in int draw_id;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out vec3 metallic_roughness_out;

// Source: https://gamedev.stackexchange.com/questions/92015/optimized-linear-to-srgb-glsl
// Converts a color from sRGB gamma to linear light gamma
vec4 toLinear(vec4 sRGB)
{
    bvec4 cutoff = lessThan(sRGB, vec4(0.04045));
    vec4 higher = pow((sRGB + vec4(0.055))/vec4(1.055), vec4(2.4));
    vec4 lower = sRGB/vec4(12.92);

    return mix(higher, lower, cutoff);
}

void main(void) {

    sampler2D base_tx_hndl = sampler2D(mesh_shader_params[draw_id].albedo_texture_handle);
    sampler2D roughness_tx_hndl = sampler2D(mesh_shader_params[draw_id].metallicRoughness_texture_handle);
    sampler2D normal_tx_hndl = sampler2D(mesh_shader_params[draw_id].normal_texture_handle);

    vec4 albedo_tx_value = texture(base_tx_hndl, uv_coord);
    vec4 roughness_tx_value = texture(roughness_tx_hndl, uv_coord);
    vec4 normal_tx_value = texture(normal_tx_hndl, uv_coord);

    bool is_sRGB = true;
    if(is_sRGB){
        albedo_tx_value = toLinear(albedo_tx_value);
        //roughness_tx_value = toLinear(roughness_tx_value);
    }

    if(albedo_tx_value.a < 0.01){
        discard;
    }

    vec2 metallicRoughness = roughness_tx_value.bg;

    vec3 tNormal = ( normal_tx_value.rgb * 2.0) - 1.0;
    normal_out.xyz = (normalize(transpose(tangent_space_matrix) * tNormal));
    
    albedo_out = albedo_tx_value;
    
    metallic_roughness_out = vec3( metallicRoughness, 0.0 );
}

#version 430
#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams {
    mat4 transform;
    uvec2 albedo_texture_handle;
    uvec2 normal_texture_handle;
    uvec2 metallicRoughness_texture_handle;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams mesh_shader_params[]; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_position;
layout(location = 2) in vec4 v_tangent;
layout(location = 3) in vec2 v_uv;

out vec3 vnormal;
out vec2 uv_coord;
out mat3 tangent_space_matrix;
flat out int draw_id;

void main()
{
    mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;

    draw_id = gl_DrawIDARB;

    /*	Construct matrices that use the model matrix*/
    mat3 normal_matrix = transpose(inverse(mat3(object_transform)));

    /*	Just to be on the safe side, normalize input vectors again */
    vec3 normal = normalize(v_normal);
    vec3 tangent = normalize(v_tangent.xyz);
    vec3 bitangent = normalize( cross(normal, tangent) * v_tangent.w );
    
    /*	Transform input vectors into view space */
    normal = normalize(normal_matrix * normal);
    tangent = normalize(normal_matrix * tangent);
    bitangent = normalize(normal_matrix * bitangent);

    /*	Compute transformation matrix for tangent space transformation */
    tangent_space_matrix = mat3(
        tangent.x, bitangent.x, normal.x,
        tangent.y, bitangent.y, normal.y,
        tangent.z, bitangent.z, normal.z);
    
    /*	Transform vertex position to view space */
    //vec3 world_pos = (object_transform * vec4(v_position,1.0)).xyz;

    vnormal = normalize(tangent).rgb;
    
    uv_coord = v_uv;

    gl_Position =  proj_mx * view_mx * object_transform * vec4(v_position,1.0);
}

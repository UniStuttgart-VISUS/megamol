#extension GL_ARB_shader_draw_parameters : require
#extension GL_ARB_bindless_texture : require

struct MeshShaderParams
{
    mat4 transform;
};

struct DynamicData
{
    int node_number;
    float node_posX;
    float node_posY;
    float node_posZ;
    float node_displX;
    float node_displY;
    float node_displZ;
    float norm_stressX;
    float norm_stressY;
    float norm_stressZ;
    float shear_stressX;
    float shear_stressY;
    float shear_stressZ;
    float padding0;
    float padding1;
    float padding2;
};

struct TextureHandle
{
    uvec2 tex_handle;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams mesh_shader_params[]; };
layout(std430, binding = 1) readonly buffer DynamicDataBuffer { DynamicData dynamic_data[]; };
layout(std430, binding = 2) readonly buffer TextureHandlesBuffer { TextureHandle texture_handles[]; };


uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec3 v_position;

out vec3 vColour;

void main()
{
    mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;
    vec3 vertex_displacement = vec3(
        dynamic_data[gl_VertexID].node_displX,
        dynamic_data[gl_VertexID].node_displY,
        dynamic_data[gl_VertexID].node_displZ
        );
    vertex_displacement *= 10.0f;
    //vertex_displacement = vec3(0.0f);

    vec3 colour = texture(sampler1D(texture_handles[0].tex_handle), dynamic_data[gl_VertexID].norm_stressY ).rgb;

    vColour = colour;

    vec3 position = vec3(
        dynamic_data[gl_VertexID].node_posX,
        dynamic_data[gl_VertexID].node_posY,
        dynamic_data[gl_VertexID].node_posZ
    );
    //position = v_position;

    gl_Position =  proj_mx * view_mx * object_transform * vec4(position + vertex_displacement,1.0);
}
#version 450

#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams
{
    mat4 transform;
    vec4 color;
    int cluster_id;
    int total_cluster_cnt;
    int padding_1;
    int padding_2;
};

struct PerFrameData
{
    int shading_mode;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };
layout(std430, binding = 1) readonly buffer PerFrameDataBuffer { PerFrameData[] per_frame_data; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) in vec3 v_position;
//layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec3 world_pos;
layout(location = 1) out vec4 color;

void main()
{
    vec4 base_color = mesh_shader_params[gl_DrawIDARB].color;

    if(per_frame_data[0].shading_mode == 0){
        color = base_color;
    }
    else {
        color = vec4(hsvSpiralColor(mesh_shader_params[gl_DrawIDARB].cluster_id, 
                                mesh_shader_params[gl_DrawIDARB].total_cluster_cnt), base_color.a);
    }

    world_pos = v_position;
    
    mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;
    gl_Position =  proj_mx * view_mx * object_transform * vec4(v_position,1.0);
    //gl_Position =  object_transform * vec4(v_position,1.0);
    //gl_Position =  vec4(v_position,1.0);
}

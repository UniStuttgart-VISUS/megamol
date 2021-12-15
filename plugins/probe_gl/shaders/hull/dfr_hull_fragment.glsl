struct PerFrameData
{
    int shading_mode;
};

layout(std430, binding = 1) readonly buffer PerFrameDataBuffer { PerFrameData[] per_frame_data; };

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) flat in int cluster_id;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;

void main(void) {
    //albedo_out = vec4(world_pos/vec3(50.0,50.0,-170.0),1.0);
    //albedo_out = vec4(0.57,0.05,0.05,1.0);
    if(per_frame_data[0].shading_mode == 0){
        albedo_out = color;
    }
    else {
        albedo_out = vec4(color_table[cluster_id % 41],1.0);
    }
    normal_out = normal;
    depth_out = gl_FragCoord.z;
}

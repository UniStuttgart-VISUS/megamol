vec3 color_table[41] = {
    vec3(0.941f, 0.973f, 1.000f),
    vec3(0.980f, 0.922f, 0.843f),
    vec3(0.000f, 1.000f, 1.000f),
    vec3(0.498f, 1.000f, 0.831f),
    vec3(0.941f, 1.000f, 1.000f),
    vec3(0.961f, 0.961f, 0.863f),
    vec3(1.000f, 0.894f, 0.769f),
    vec3(0.000f, 0.000f, 0.000f),
    vec3(1.000f, 0.922f, 0.804f),
    vec3(0.000f, 0.000f, 1.000f),
    vec3(0.541f, 0.169f, 0.886f),
    vec3(0.647f, 0.165f, 0.165f),
    vec3(0.871f, 0.722f, 0.529f),
    vec3(0.373f, 0.620f, 0.627f),
    vec3(0.498f, 1.000f, 0.000f),
    vec3(0.824f, 0.412f, 0.118f),
    vec3(1.000f, 0.498f, 0.314f),
    vec3(0.392f, 0.584f, 0.929f),
    vec3(1.000f, 0.973f, 0.863f),
    vec3(0.863f, 0.078f, 0.235f),
    vec3(0.000f, 1.000f, 1.000f),
    vec3(0.000f, 0.000f, 0.545f),
    vec3(0.000f, 0.545f, 0.545f),
    vec3(0.722f, 0.525f, 0.043f),
    vec3(0.663f, 0.663f, 0.663f),
    vec3(0.000f, 0.392f, 0.000f),
    vec3(0.663f, 0.663f, 0.663f),
    vec3(0.741f, 0.718f, 0.420f),
    vec3(0.545f, 0.000f, 0.545f),
    vec3(0.333f, 0.420f, 0.184f),
    vec3(1.000f, 0.549f, 0.000f),
    vec3(0.600f, 0.196f, 0.800f),
    vec3(0.545f, 0.000f, 0.000f),
    vec3(0.914f, 0.588f, 0.478f),
    vec3(0.561f, 0.737f, 0.561f),
    vec3(0.282f, 0.239f, 0.545f),
    vec3(0.184f, 0.310f, 0.310f),
    vec3(0.184f, 0.310f, 0.310f),
    vec3(0.000f, 0.808f, 0.820f),
    vec3(0.580f, 0.000f, 0.827f),
    vec3(1.000f, 0.078f, 0.576f)
};



layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

layout(std430, binding = 1) readonly buffer PerFrameDataBuffer { PerFrameData[] per_frame_data; };

uniform mat4 view_mx;

layout(location = 0) flat in int draw_id;
layout(location = 1) in vec2 uv_coords;
layout(location = 2) in vec3 pixel_vector;
layout(location = 3) in vec3 cam_vector;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;
layout(location = 3) out int objID_out;
layout(location = 4) out vec4 interactionData_out;

#define PI 3.1415926

vec3 projectOntoPlane(vec3 v, vec3 n)
{
    return ( v - (( dot(v,n) / length(n) ) * n) );
};

void main() {

    if(dot(cam_vector,mesh_shader_params[draw_id].probe_direction.xyz) < 0.0 ){
        discard;
    }

    vec4 glyph_border_color = vec4(1.0);

    if(mesh_shader_params[draw_id].state == 1) {
        glyph_border_color = vec4(1.0,1.0,0.0,1.0);
    }
    else if(mesh_shader_params[draw_id].state == 2) {
        glyph_border_color = vec4(1.0,0.58,0.0,1.0);
    }

    // Highlight glyph up and glyph right directions
    if( (uv_coords.x > 0.99 && uv_coords.x > uv_coords.y && uv_coords.y > 0.9) ||
        (uv_coords.y > 0.99 && uv_coords.x < uv_coords.y && uv_coords.x > 0.9) ||
        (uv_coords.x < 0.01 && uv_coords.x < uv_coords.y && uv_coords.y < 0.05) ||
        (uv_coords.y < 0.01 && uv_coords.x > uv_coords.y && uv_coords.x < 0.05) )
    {
        albedo_out = glyph_border_color;
        normal_out = vec3(0.0,0.0,1.0);
        depth_out = gl_FragCoord.z;
        objID_out = mesh_shader_params[draw_id].probe_id;
        return;
    }
    
    float r = length(uv_coords - vec2(0.5)) * 2.0;

    if(r > 1.0) discard;
    if(r < 0.1) discard;
    
    vec3 out_colour = color_table[mesh_shader_params[draw_id].cluster_id % 41];

    albedo_out = vec4(out_colour,1.0);
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;

    objID_out = mesh_shader_params[draw_id].probe_id;
    interactionData_out = vec4(0.0);
}
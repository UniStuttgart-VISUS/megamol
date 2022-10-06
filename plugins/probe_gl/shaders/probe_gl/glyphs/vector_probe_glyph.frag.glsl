#version 450

#include "probe_gl/glyphs/extensions.inc.glsl"
#include "probe_gl/glyphs/per_frame_data_struct.inc.glsl"
#include "probe_gl/glyphs/vector_probe_struct.inc.glsl"
#include "probe_gl/glyphs/dial_glyph_utility.inc.glsl"

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
    if(highlightCorners(uv_coords))
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

    float pixel_diag_width = 1.5 * max(dFdx(uv_coords.x),dFdy(uv_coords.y));

    float border_circle_width = 0.02;
    if(mesh_shader_params[draw_id].state == 1) {
        border_circle_width = 0.06;
    }
    else if(mesh_shader_params[draw_id].state == 2) {
        border_circle_width = 0.06;
    }
    border_circle_width = max(border_circle_width,pixel_diag_width);

    if(r > (1.0 - border_circle_width)){
        albedo_out = glyph_border_color;
        normal_out = vec3(0.0,0.0,1.0);
        depth_out = gl_FragCoord.z;
        objID_out = mesh_shader_params[draw_id].probe_id;
        return;
    }

    float radar_sections_cnt = mesh_shader_params[draw_id].sample_cnt;
    vec3 proj_pv = normalize(projectOntoPlane(pixel_vector,mesh_shader_params[draw_id].probe_direction.xyz));
    
    vec3 out_colour = per_frame_data[0].canvas_color.rgb;
    bool interpolate = bool(per_frame_data[0].use_interpolation);

    vec3 sample_vector = vec3(0.0);
    float sample_magnitude = 0.0;

    // inverse direction of sample lookup to map higher sample depth to smaller radius
    // also shift slightly away from probe center, since that region is not useful to clearly show directions
    float invere_r = 1.0 - ( (r - 0.1) / 0.9 );

    if(interpolate)
    {    
        // identify section of radar glyph that the pixel belongs to
        int radar_section_0 = clamp(int(floor(invere_r * (radar_sections_cnt - 1.0))),0,int(radar_sections_cnt)-1);
        int radar_section_1 = clamp(int(ceil(invere_r * (radar_sections_cnt - 1.0))),0,int(radar_sections_cnt)-1);
        float lerp = fract(invere_r * (radar_sections_cnt-1));

        // based on section, calculate vector projection
        vec3 sample_vector_0 = normalize(mesh_shader_params[draw_id].samples[radar_section_0].xyz);
        float sample_magnitude_0 = mesh_shader_params[draw_id].samples[radar_section_0].w;

        vec3 sample_vector_1 = normalize(mesh_shader_params[draw_id].samples[radar_section_1].xyz);
        float sample_magnitude_1 = mesh_shader_params[draw_id].samples[radar_section_1].w;

        sample_vector = normalize(mix(sample_vector_0,sample_vector_1,lerp));
        sample_magnitude = mix(sample_magnitude_0,sample_magnitude_1,lerp);
    }
    else
    {
        int radar_section = clamp(int(round(invere_r * (radar_sections_cnt - 1.0))),0,int(radar_sections_cnt)-1);

        sample_vector = normalize(mesh_shader_params[draw_id].samples[radar_section].xyz);
        sample_magnitude = mesh_shader_params[draw_id].samples[radar_section].w;
    }

    bool sample_is_valid = ( !isnan(sample_magnitude) && (sample_magnitude > 0.005 || sample_magnitude < -0.005));
        //discard invalid samples
        //if( isnan(sample_magnitude) ) discard;
        //if(sample_magnitude < 0.005 && sample_magnitude > -0.005) discard;

    if(sample_is_valid){
        vec3 proj_sv =  normalize(projectOntoPlane(sample_vector,mesh_shader_params[draw_id].probe_direction.xyz));

        float proj_sample_dot_pixel = dot(proj_sv,proj_pv);
        float sample_dot_probe = dot(sample_vector,mesh_shader_params[draw_id].probe_direction.xyz);

        float circumference = 2.0 * PI * r;
        float inner_angle = acos(proj_sample_dot_pixel);

        float arc_length = (inner_angle / (2.0*PI)) * 2.0 * circumference;
        float tgt_arc_length = ( 1.0 - (acos(abs(sample_dot_probe)) / (0.5*PI)) ) * circumference;

        float eps = -max(0.05,pixel_diag_width);
        if( (arc_length + eps) < tgt_arc_length ){
            sampler2D tf_tx = sampler2D(per_frame_data[0].tf_texture_handle);
            float tf_min = per_frame_data[0].tf_min;
            float tf_max = per_frame_data[0].tf_max;
            out_colour = texture(tf_tx, vec2((sample_magnitude - tf_min) / (tf_max-tf_min), 0.5) ).rgb;
            //out_colour = fakeViridis( (sample_magnitude + 2.0) / 16.0);
        }
        else{
            if(per_frame_data[0].show_canvas == 0) discard;
        }
    }
    else{
        if(per_frame_data[0].show_canvas == 0) discard;
    }

    albedo_out = vec4(out_colour,1.0);
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;

    objID_out = mesh_shader_params[draw_id].probe_id;
    interactionData_out = vec4(0.0);
}

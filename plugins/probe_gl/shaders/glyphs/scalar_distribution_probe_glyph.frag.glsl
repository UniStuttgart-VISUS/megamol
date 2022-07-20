#version 450

#include "glyphs/extensions.inc.glsl"
#include "glyphs/scalar_distribution_probe_struct.inc.glsl"

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

layout(std430, binding = 1) readonly buffer PerFrameDataBuffer { PerFrameData[] per_frame_data; };

layout(location = 0) flat in int draw_id;
layout(location = 1) in vec2 uv_coords;
layout(location = 2) in vec3 pixel_vector;
layout(location = 3) in vec3 cam_vector;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;
layout(location = 3) out int objID_out;
layout(location = 4) out vec4 interactionData_out;

vec3 fakeViridis(float lerp)
{
    vec3 c0 = vec3(0.2823645529290169,0.0,0.3310101940118055);
    vec3 c1 = vec3(0.24090172204161298,0.7633448774061599,0.42216355577803744);
    vec3 c2 = vec3(0.9529994532916154,0.9125452328290099,0.11085876909361342);

    return lerp < 0.5 ? mix(c0,c1,lerp * 2.0) : mix(c1,c2,(lerp*2.0)-1.0);
};

void main() {

    if(dot(cam_vector,mesh_shader_params[draw_id].probe_direction.xyz) < 0.0 ){
        discard;
    }

    vec4 glyph_border_color = vec4(0.0,0.0,0.0,1.0);

    if(mesh_shader_params[draw_id].state == 1) {
        glyph_border_color = vec4(1.0,1.0,0.0,1.0);
    }
    else if(mesh_shader_params[draw_id].state == 2) {
        glyph_border_color = vec4(1.0,0.58,0.0,1.0);
    }

    // Highlight glyph up and glyph right directions
    //if( (uv_coords.x > 0.99 && uv_coords.x > uv_coords.y && uv_coords.y > 0.9) ||
    //    (uv_coords.y > 0.99 && uv_coords.x < uv_coords.y && uv_coords.x > 0.9) ||
    //    (uv_coords.x < 0.01 && uv_coords.x < uv_coords.y && uv_coords.y < 0.05) ||
    //    (uv_coords.y < 0.01 && uv_coords.x > uv_coords.y && uv_coords.x < 0.05) )
    //{
    //    albedo_out = glyph_border_color;
    //    normal_out = vec3(0.0,0.0,1.0);
    //    depth_out = gl_FragCoord.z;
    //    objID_out = mesh_shader_params[draw_id].probe_id;
    //    return;
    //}

    vec2 pixel_coords = uv_coords * 2.0 - vec2(1.0,1.0);
    float radius = length(pixel_coords);

    if(radius > 1.0) discard;

    float pixel_diag_width = 1.5 * max(dFdx(uv_coords.x),dFdy(uv_coords.y));

    float border_circle_width = 0.04;
    if(mesh_shader_params[draw_id].state == 1) {
        border_circle_width = 0.08;
    }
    else if(mesh_shader_params[draw_id].state == 2) {
        border_circle_width = 0.08;
    }
    border_circle_width = 2.0 * max(border_circle_width,pixel_diag_width);

    float angle = atan(
        pixel_coords.x,
        pixel_coords.x > 0.0 ? -pixel_coords.y : pixel_coords.y
    );

    if(pixel_coords.x < 0.0){
        angle = angle * -1.0 + 3.14159;
    }

    float angle_normalized = angle / (3.14159*2.0);
    angle_normalized = 1.0 - angle_normalized; // invert for clockwise reading

    vec3 out_colour = vec3(0.0,0.0,0.0);

    float min_value = per_frame_data[0].tf_min;
    float max_value = per_frame_data[0].tf_max;
    float value_range = max_value - min_value;

    float zero_value_radius = -min_value / value_range;
    float zero_arc_width = max(0.005, 0.5 * pixel_diag_width);

    if(angle_normalized > 0.025 && angle_normalized < 0.975 && (radius > zero_value_radius+zero_arc_width || radius < zero_value_radius-zero_arc_width) )
    {
        float angle_shifted = (angle_normalized - 0.025) / 0.95;

        int sample_cnt = int(mesh_shader_params[draw_id].sample_cnt);
        int sample_idx_0 = int(floor(angle_shifted * sample_cnt));
        int sample_idx_1 = int(ceil(angle_shifted * sample_cnt));
        float lerp = fract(angle_shifted * sample_cnt);

        float sample_0 = mesh_shader_params[draw_id].samples[sample_idx_0].x;
        float sample_1 = mesh_shader_params[draw_id].samples[sample_idx_1].x;

        if( isnan(sample_0) || isnan(sample_1)) discard;

        sampler2D tf_tx = sampler2D(per_frame_data[0].tf_texture_handle);

        bool interpolate = bool(per_frame_data[0].use_interpolation);
        
        float sample_mean_value_normalized = 0.0;
        float sample_lower_value_normalized = 0.0;
        float sample_upper_value_normalized = 0.0;

        //if(interpolate)
        //{
        //    float sample_value = mix(sample_0,sample_1,lerp);
        //    sample_value_normalized = (sample_value - min_value) / (value_range);
        //    //out_colour = fakeViridis(sample_value_normalized);
        //    //out_colour = texture(tf_tx, vec2(sample_value_normalized, 1.0) ).rgb;
        //    //if( radius > sample_value_normalized && radius < border_circle_width ) discard;
        //}
        //else
        {
            int sample_idx = int(round(angle_shifted * sample_cnt));
            float sample_mean_value = mesh_shader_params[draw_id].samples[sample_idx].x;
            float sample_lower_value = mesh_shader_params[draw_id].samples[sample_idx].y;
            float sample_upper_value = mesh_shader_params[draw_id].samples[sample_idx].z;
            sample_mean_value_normalized = (sample_mean_value - min_value) / (value_range);
            sample_lower_value_normalized = (sample_lower_value - min_value) / (value_range);
            sample_upper_value_normalized = (sample_upper_value - min_value) / (value_range);
            //out_colour = fakeViridis(sample_value_normalized);
            //out_colour = texture(tf_tx, vec2(sample_value_normalized, 1.0) ).rgb;
            //if( radius > sample_value_normalized && radius < border_circle_width ) discard;
        }

        out_colour = texture(tf_tx, vec2(sample_mean_value_normalized, 1.0) ).rgb;

        if( (radius > sample_upper_value_normalized || radius < sample_lower_value_normalized) && radius < (1.0 - border_circle_width) ){
            if( abs(radius - zero_value_radius) > 0.005 && abs(radius - sample_mean_value_normalized) > 0.02 ){
                if(bool(per_frame_data[0].show_canvas)){
                    out_colour = per_frame_data[0].canvas_color.rgb;
                }
                else{
                    discard;
                }
            } 
        }
        
        if( (radius < sample_upper_value_normalized) && (radius > sample_lower_value_normalized) && radius < (1.0 - border_circle_width) ){
            if(abs(radius - sample_mean_value_normalized) > 0.02) out_colour = vec3(0.3);
        }
    }

    if(abs(radius - zero_value_radius) < zero_arc_width) out_colour = vec3(1.0);
    if(radius > (1.0 - border_circle_width) && radius < (1.0 - (0.5*border_circle_width))) out_colour = glyph_border_color.rgb;

    float test = dFdx(radius);

    albedo_out = vec4(out_colour,1.0);
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;

    objID_out = mesh_shader_params[draw_id].probe_id;
    interactionData_out = vec4(0.0);
}

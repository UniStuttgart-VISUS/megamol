#version 450

#include "probe_gl/glyphs/extensions.inc.glsl"
#include "probe_gl/glyphs/per_frame_data_struct.inc.glsl"
#include "probe_gl/glyphs/base_probe_struct.inc.glsl"
#include "probe_gl/glyphs/dial_glyph_constants.inc.glsl"

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

layout(std430, binding = 1) readonly buffer PerFrameDataBuffer { PerFrameData[] per_frame_data; };

layout(location = 0) flat in int draw_id;
layout(location = 1) in vec2 uv_coords;
layout(location = 2) in vec3 pixel_vector;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;
layout(location = 3) out int objID_out;
layout(location = 4) out vec4 interactionData_out;

void main() {
    vec2 pixel_coords = uv_coords * 2.0 - vec2(1.0,1.0);
    float radius = length(pixel_coords);

    if(radius > 1.0){
        discard;
    }

    vec2 pixel_direction = normalize(pixel_coords);
    float angle = atan(
        pixel_direction.x,
        pixel_direction.x > 0.0 ? -pixel_direction.y : pixel_direction.y
    );
    angle = pixel_coords.x < 0.0 ? angle * -1.0 + 3.14159 : angle;
    float angle_normalized = 1.0 - (angle/6.283185 /*2pi*/);

    float pixel_diag_width = 1.5 * max(dFdx(uv_coords.x),dFdy(uv_coords.y));
    float border_line_width = max(base_line_width, pixel_diag_width);

    if(mesh_shader_params[draw_id].state == 1) {
        border_line_width *= 2.0;
    }
    else if(mesh_shader_params[draw_id].state == 2) {
        border_line_width *= 2.0;
    }

    float angle_line_width = border_line_width / (6.283185 * radius);

    float arrow_radius = angle_normalized < angle_arrow_start ? arrow_base_radius : arrow_base_radius + pow(angle_normalized-angle_arrow_start,4.0) * 20.0;
    float arrow_line_width = angle_normalized < angle_arrow_start ? border_line_width : border_line_width + border_line_width*(1.0-2.0*smoothstep(angle_arrow_start,angle_end,angle_normalized));
    arrow_line_width *= 0.5;

    bool radius_is_outer_border_circle = radius > (1.0 - border_line_width);
    bool radius_is_plot_draw_area = (radius < (1.0 - border_line_width)) && (radius > (inner_radius + border_line_width));
    bool radius_is_inner_border_circle = (radius < (inner_radius + border_line_width)) && (radius > inner_radius);
    bool radius_is_arrow = (radius < arrow_radius+arrow_line_width && radius > arrow_radius-arrow_line_width);

    if(angle_normalized < angle_start || angle_normalized > angle_end){
        discard;
    }
    else if(angle_normalized < angle_start+angle_line_width){
        if(!(radius_is_outer_border_circle || radius_is_inner_border_circle || radius_is_arrow) && !radius_is_plot_draw_area){
            discard;
        }
    }
    else if(angle_normalized < angle_arrow_start){
        if(!(radius_is_outer_border_circle || radius_is_inner_border_circle || radius_is_arrow)){
            discard;
        }
    }
    else if(angle_normalized < angle_end-angle_line_width){
        if(!(radius_is_outer_border_circle || radius_is_inner_border_circle || radius_is_arrow)){
            discard;
        }
    }
    else{
        if(!(radius_is_outer_border_circle || radius_is_inner_border_circle || radius_is_arrow) && !radius_is_plot_draw_area){
                discard;
            }
    }

    vec4 out_colour = vec4(0.0,0.0,0.0,1.0);

    if(mesh_shader_params[draw_id].state == 1) {
        out_colour = vec4(1.0,1.0,0.0,1.0);
    }
    else if(mesh_shader_params[draw_id].state == 2) {
        out_colour = vec4(1.0,0.58,0.0,1.0);
    }

    albedo_out = out_colour;
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;

    objID_out = mesh_shader_params[draw_id].probe_id;
    interactionData_out = vec4(0.0);
}

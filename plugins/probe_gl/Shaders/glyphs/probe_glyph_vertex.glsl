
layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) flat out int draw_id;
layout(location = 1) out vec2 uv_coords;
layout(location = 2) out vec3 pixel_vector;

void main()
{
    const vec4 vertices[6] = vec4[6]( vec4( -1.0,-1.0,0.0,0.0 ),
									  vec4( 1.0,1.0,1.0,1.0 ),
									  vec4( -1.0,1.0,0.0,1.0 ),
									  vec4( 1.0,1.0,1.0,1.0 ),
									  vec4( -1.0,-1.0,0.0,0.0 ),
                                	  vec4( 1.0,-1.0,1.0,0.0 ) );

    draw_id = gl_DrawIDARB;
    uv_coords = vertices[gl_VertexID].zw;

    vec4 glyph_pos = mesh_shader_params[gl_DrawIDARB].glpyh_position;
    vec4 clip_pos = (proj_mx * view_mx * glyph_pos);  
    float aspect = proj_mx[1][1] / proj_mx[0][0];
    vec2  bboard_vertex = vertices[gl_VertexID].xy;

    ////
    // compute plausible glyph up vector orthognal to probe direction
    vec3 probe_direction = mesh_shader_params[draw_id].probe_direction.xyz;

    vec3 cam_right = normalize(transpose(mat3(view_mx)) * vec3(1.0,0.0,0.0));
    vec3 cam_front = normalize(transpose(mat3(view_mx)) * vec3(0.0,0.0,-1.0));

    float cr_dot_pd = dot(cam_right,probe_direction);
    float cf_dot_pd = dot(cam_front,probe_direction);

    vec3 glyph_up_0 = normalize(cross(probe_direction,cam_right)) * -sign(cf_dot_pd);;
    vec3 glyph_up_1 = normalize(cross(probe_direction,cam_front)) * sign(cr_dot_pd);;

    vec3 glyph_right_0 = normalize(cross(probe_direction,glyph_up_0));
    vec3 glyph_right_1 = normalize(cross(probe_direction,glyph_up_1));

    vec2 pixel_coords = uv_coords * 2.0 - 1.0;

    if( abs(cf_dot_pd) < 0.9 && abs(cf_dot_pd) > 0.25 )
    {
        vec3 glyph_up = normalize( mix(glyph_up_0,glyph_up_1, 1.0 - ( abs(cf_dot_pd) ) ) );
        vec3 glyph_right = normalize( mix(glyph_right_0,glyph_right_1, 1.0 - ( abs(cf_dot_pd) ) ) );

        pixel_vector = normalize( pixel_coords.x * glyph_right + pixel_coords.y * glyph_up );

        //pixel_vector = pixel_coords.x * glyph_right;

        //pixel_vector = vec3(1.0,0.0,1.0);
    }
    else if(abs(cr_dot_pd) < abs(cf_dot_pd))
    {
        pixel_vector = normalize( pixel_coords.x * glyph_right_0 + pixel_coords.y * glyph_up_0 );

        //pixel_vector = pixel_coords.x * glyph_right_0;

        //pixel_vector = vec3(1.0,0.0,1.0);
    }
    else
    {
        pixel_vector = normalize( pixel_coords.x * glyph_right_1 + pixel_coords.y * glyph_up_1 );

        //pixel_vector = pixel_coords.x * glyph_right_1;

        //pixel_vector = vec3(1.0,0.0,1.0);
    }

    //pixel_vector = vec3(dot(glyph_right_0,glyph_right_1),0.0,0.0);
    
    gl_Position = clip_pos + vec4(bboard_vertex.x * mesh_shader_params[gl_DrawIDARB].scale,
                                 bboard_vertex.y * mesh_shader_params[gl_DrawIDARB].scale * aspect, 0.0, 0.0);
}
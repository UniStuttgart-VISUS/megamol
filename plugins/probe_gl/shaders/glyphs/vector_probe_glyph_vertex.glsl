
layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) flat out int draw_id;
layout(location = 1) out vec2 uv_coords;
layout(location = 2) out vec3 pixel_vector;
layout(location = 3) out vec3 cam_vector;

//http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

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

    vec4 glyph_pos = vec4(mesh_shader_params[gl_DrawIDARB].glpyh_position.xyz, 1.0);
    vec4 clip_pos = (proj_mx * view_mx * glyph_pos);  
    float aspect = proj_mx[1][1] / proj_mx[0][0];
    vec2  bboard_vertex = vertices[gl_VertexID].xy;

    ////
    // compute plausible glyph up vector orthognal to probe direction
    vec3 probe_direction = normalize( mesh_shader_params[draw_id].probe_direction.xyz );
    vec3 glyph_up = vec3(0.0,0.0,0.0);
    vec3 glyph_right = vec3(0.0,0.0,0.0);

    // identify coord axis that the probe is most parallel to
    if( abs(dot(probe_direction,vec3(0.0,0.0,1.0))) > (sqrt(2)/2.0) )
    {
        glyph_up = normalize(cross(probe_direction,vec3(1.0,0.0,0.0))) * sign(dot(probe_direction,vec3(0.0,0.0,1.0)));
        glyph_right = normalize(cross(probe_direction,glyph_up));
        glyph_up = normalize(cross(glyph_right,probe_direction));
    }
    else if(abs(dot(probe_direction,vec3(1.0,0.0,0.0))) > (sqrt(2)/2.0))
    {
        glyph_up = normalize(cross(vec3(0.0,0.0,1.0),probe_direction)) * sign(dot(probe_direction,vec3(1.0,0.0,0.0)));
        glyph_right = normalize(cross(probe_direction,glyph_up));
        glyph_up = normalize(cross(glyph_right,probe_direction));
    }
    else
    {
        glyph_up = normalize(cross(vec3(1.0,0.0,0.0),probe_direction)) * sign(dot(probe_direction,vec3(0.0,1.0,0.0)));
        glyph_right = normalize(cross(probe_direction,glyph_up));
        glyph_up = normalize(cross(glyph_right,probe_direction));
    }

    vec2 pixel_coords = uv_coords * 2.0 - 1.0;
    pixel_vector = normalize( pixel_coords.x * glyph_right + pixel_coords.y * glyph_up );

    // tilt glyph towards camera a little bit
    vec3 cam_front = normalize(transpose(mat3(view_mx)) * vec3(0.0,0.0,-1.0));
    cam_vector = cam_front;

    float probe_dot_cam = dot(probe_direction, cam_front);
    if( probe_dot_cam > 0.0 )
    {
        float angle = probe_dot_cam;
        vec3 axis = normalize(cross(probe_direction, cam_front));

        mat4 rot = rotationMatrix(axis,-acos(angle));

        vec3 tilted_glyph_up = (rot * vec4(glyph_up,1.0)).xyz;
        vec3 tilted_glyph_right = (rot * vec4(glyph_right,1.0)).xyz;

        float tilt_factor = (acos(probe_dot_cam) / 1.57);
        tilt_factor = pow(tilt_factor,4.0);
        glyph_up = normalize(mix(tilted_glyph_up, glyph_up, tilt_factor));
        glyph_right = normalize(mix(tilted_glyph_right, glyph_right, tilt_factor));
    }

    glyph_pos.xyz = glyph_pos.xyz + (glyph_up * bboard_vertex.y * mesh_shader_params[gl_DrawIDARB].scale) + (glyph_right * bboard_vertex.x * mesh_shader_params[gl_DrawIDARB].scale);

    gl_Position = (proj_mx * view_mx * glyph_pos);

    {
        /*
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
            glyph_up = normalize( mix(glyph_up_0,glyph_up_1, 1.0 - (cf_dot_pd * 0.5 + 0.5) ) );
            glyph_right = normalize( mix(glyph_right_0,glyph_right_1, 1.0 - (cf_dot_pd * 0.5 + 0.5) ) );

            //pixel_vector = pixel_coords.x * glyph_right;
            //pixel_vector = vec3(1.0,0.0,1.0);
        }
        else if(abs(cr_dot_pd) < abs(cf_dot_pd))
        {
            glyph_up = glyph_up_0;
            glyph_right = glyph_right_0;

            //pixel_vector = pixel_coords.x * glyph_right_0;
            //pixel_vector = vec3(1.0,0.0,1.0);
        }
        else
        {
            glyph_up = glyph_up_1;
            glyph_right = glyph_right_1;

            //pixel_vector = pixel_coords.x * glyph_right_1;
            //pixel_vector = vec3(1.0,0.0,1.0);
        }

        pixel_vector = normalize( pixel_coords.x * glyph_right + pixel_coords.y * glyph_up );
        //pixel_vector = vec3(dot(glyph_right_0,glyph_right_1),0.0,0.0);
        */
    }
    
    //gl_Position = clip_pos + vec4(bboard_vertex.x * mesh_shader_params[gl_DrawIDARB].scale,
    //                             bboard_vertex.y * mesh_shader_params[gl_DrawIDARB].scale * aspect, 0.0, 0.0);
}
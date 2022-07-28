#version 450

uniform mat4 view_mx;

in vec3 world_pos;
in vec3 normal;
in float force;

out layout(location = 0) vec4 frag_colour;

vec3 lighting(vec3 view_direction, vec3 light_direction, vec3 normal)
{
    vec3 halfway = normalize(light_direction + view_direction);
    float l_dot_h = dot(light_direction,halfway);
    float v_dot_n = clamp(dot(view_direction,normal),0.0,1.0);
    float n_dot_h = clamp(dot( normal, halfway ),0.0,1.0);

    /*
    /  Compute Fresnel term using the Schlick approximation.
    /  To avoid artefacts, a small epsilon is added to 1.0-l_dot_h
    */
    float fresnel = mix(0.2, 1.0, pow(1.01-l_dot_h,5.0));

    float viewDependentRoughness = mix(1.0, 0.3, pow(1.01-v_dot_n,5.0));

    float lambert = max(0.0,dot(light_direction, normal));

    float blinnPhong = pow(n_dot_h, 22.0 );
    blinnPhong = ceil(lambert);

    return vec3(1.0) * mix(blinnPhong,lambert,viewDependentRoughness);
}

void main(void) {

    vec3 view_direction = normalize( (inverse(view_mx) * vec4(0.0,0.0,0.0,1.0)).xyz - world_pos );
    vec3 light_direction_0 = normalize(vec3(2.0,3.0,2.0) - world_pos);
    vec3 light_direction_1 = normalize(vec3(2.0,3.0,-2.0) - world_pos);
    vec3 light_direction_2 = normalize(vec3(-2.0,3.0,2.0) - world_pos);
    vec3 light_direction_3 = normalize(vec3(-2.0,3.0,-2.0) - world_pos);
    
    vec3 out_color = 0.25 * lighting(view_direction,light_direction_0,normal);
    out_color += 0.25 * lighting(view_direction,light_direction_1,normal);
    out_color += 0.25 * lighting(view_direction,light_direction_2,normal);
    out_color += 0.25 * lighting(view_direction,light_direction_3,normal);

    //out_color = mix(vec3(0.0,0.0,1.0),vec3(1.0,0.0,0.0), (force+100.0)/200.0);
    if(force < 0.0)
    {
        out_color = mix(out_color,vec3(0.0,0.0,1.0),abs(force)/100.0);
    }
    else
    {
    out_color = mix(out_color,vec3(1.0,0.0,0.0),force/100.0);
    }

    frag_colour = vec4( out_color, 1.0);
}

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

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

vec3 fakeViridis(float lerp)
{
    vec3 c0 = vec3(0.2823645529290169,0.0,0.3310101940118055);
    vec3 c1 = vec3(0.24090172204161298,0.7633448774061599,0.42216355577803744);
    vec3 c2 = vec3(0.9529994532916154,0.9125452328290099,0.11085876909361342);

    return lerp < 0.5 ? mix(c0,c1,lerp * 2.0) : mix(c1,c2,(lerp*2.0)-1.0);
};

vec3 projectOntoPlane(vec3 v, vec3 n)
{
    return ( v - (( dot(v,n) / length(n) ) * n) );
};

void main() {

    if(dot(cam_vector,mesh_shader_params[draw_id].probe_direction.xyz) < 0.0 ){
        discard;
    }

    // For debugging purposes, hightlight glyph up and glyph right directions
    if(uv_coords.x > 0.99 && uv_coords.x > uv_coords.y && uv_coords.y > 0.9) 
    {
        albedo_out = vec4(1.0);
        normal_out = vec3(0.0,0.0,1.0);
        depth_out = gl_FragCoord.z;
        return;
    }
    else if(uv_coords.y > 0.99 && uv_coords.x < uv_coords.y && uv_coords.x > 0.9)
    {
        albedo_out = vec4(1.0);
        normal_out = vec3(0.0,0.0,1.0);
        depth_out = gl_FragCoord.z;
        return;
    }
    else if(uv_coords.x < 0.01 && uv_coords.x < uv_coords.y && uv_coords.y < 0.05)
    {
        albedo_out = vec4(1.0);
        normal_out = vec3(0.0,0.0,1.0);
        depth_out = gl_FragCoord.z;
        return;
    }
    else if(uv_coords.y < 0.01 && uv_coords.x > uv_coords.y && uv_coords.x < 0.05)
    {
        albedo_out = vec4(1.0);
        normal_out = vec3(0.0,0.0,1.0);
        depth_out = gl_FragCoord.z;
        return;
    }

    float radar_sections_cnt = mesh_shader_params[draw_id].sample_cnt;
    float r = length(uv_coords - vec2(0.5)) * 2.0;
    //vec2 pixel_vector = normalize(uv_coords - vec2(0.5));

    //vec3 pixel_vector_ws = normalize(transpose(mat3(view_mx)) * vec3(pixel_vector,0.0));
    vec3 proj_pv = normalize(projectOntoPlane(pixel_vector,mesh_shader_params[draw_id].probe_direction.xyz));
    float pixel_dot_probe = dot(pixel_vector,mesh_shader_params[draw_id].probe_direction.xyz);

    if(r > 1.0) discard;

	// inverse direction of sample lookup to map higher sample depth to smaller radius
	r = 1.0 - r;

    // identify section of radar glyph that the pixel belongs to
    int radar_section_0 = int(floor(r * radar_sections_cnt));
    int radar_section_1 = int(ceil(r * radar_sections_cnt));
    float lerp = fract(r * radar_sections_cnt);

    // based on section, calculate vector projection
    vec3 sample_vector_0 = normalize(mesh_shader_params[draw_id].samples[radar_section_0].xyz);
    float sample_magnitude_0 = mesh_shader_params[draw_id].samples[radar_section_0].w;

    vec3 sample_vector_1 = normalize(mesh_shader_params[draw_id].samples[radar_section_1].xyz);
    float sample_magnitude_1 = mesh_shader_params[draw_id].samples[radar_section_1].w;

    vec3 out_colour = vec3(0.0,0.0,0.0);
    bool interpolate = true;

    if(interpolate){
        vec3 sample_vector = mix(sample_vector_0,sample_vector_1,lerp);
        float sample_magnitude = mix(sample_magnitude_0,sample_magnitude_1,lerp);

        //vec2 proj = normalize(sample_vector.xy);
        vec3 proj = normalize(projectOntoPlane(sample_vector,mesh_shader_params[draw_id].probe_direction.xyz));

        float sample_dot_probe = dot(sample_vector,mesh_shader_params[draw_id].probe_direction.xyz);
        float sample_dot_pixel = dot(sample_vector,proj_pv);

        float arc_dist = sample_dot_pixel * -0.5 + 0.5;

        float diff0 = sample_dot_probe - pixel_dot_probe;
        float diff1 = sample_dot_probe - arc_dist;

        float eps = -0.05;
        if( (eps + arc_dist) > abs( sample_dot_probe ) ) discard;

        sampler2D tf_tx = sampler2D(mesh_shader_params[draw_id].tf_texture_handle);
        float tf_min = mesh_shader_params[draw_id].tf_min;
        float tf_max = mesh_shader_params[draw_id].tf_max;
        out_colour = texture(tf_tx, vec2((sample_magnitude - tf_min) / (tf_max-tf_min), 0.5) ).rgb;
        //out_colour = fakeViridis( (sample_magnitude + 2.0) / 16.0);
        
    }
    //else{
    //    // for now, try projection onto z-plane for billboards
    //    vec2 proj = normalize(sample_vector_0.xy);
    //    float arc_dist = (dot(proj,pixel_vector) * -0.5) + 0.5;
//
    //    if(arc_dist > abs(sample_vector_0.z)) discard;
//
    //    out_colour = fakeViridis(sample_magnitude_0 / 2.0);
    //}

    if(mesh_shader_params[draw_id].state == 1)
    {
        out_colour = vec3(1.0,0.0,1.0);
    }

    albedo_out = vec4(out_colour,1.0);
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;

    objID_out = mesh_shader_params[draw_id].probe_id;
    interactionData_out = vec4(0.0);
}
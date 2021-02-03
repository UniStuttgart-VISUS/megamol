
layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

layout(location = 0) flat in int draw_id;
layout(location = 1) in vec2 uv_coords;

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

    vec2 pixel_coords = uv_coords * 2.0 - vec2(1.0,1.0);
    float radius = length(pixel_coords);

    if(radius > 1.0) discard;

    float angle = atan(
        pixel_coords.x,
        pixel_coords.x > 0.0 ? -pixel_coords.y : pixel_coords.y
    );

    if(pixel_coords.x < 0.0){
        angle = angle * -1.0 + 3.14159;
    }

    float angle_normalized = angle / (3.14159*2.0);

    vec3 out_colour = vec3(0.0,0.0,0.0);

    if(angle_normalized > 0.025 && angle_normalized < 0.975)
    {
        float angle_shifted = (angle_normalized - 0.025) / 0.95;

        int sample_cnt = int(mesh_shader_params[draw_id].sample_cnt);
        int sample_idx_0 = int(floor(angle_shifted * sample_cnt));
        int sample_idx_1 = int(ceil(angle_shifted * sample_cnt));
        float lerp = fract(angle_shifted * sample_cnt);

        float sample_0 = mesh_shader_params[draw_id].samples[sample_idx_0];
        float sample_1 = mesh_shader_params[draw_id].samples[sample_idx_1];

        float sample_value = mix(sample_0,sample_1,lerp);

        float sample_value_normalized = (sample_value - mesh_shader_params[draw_id].min_value) / (mesh_shader_params[draw_id].max_value - mesh_shader_params[draw_id].min_value);
        //out_colour = fakeViridis(sample_value_normalized);
		sampler2D tf_tx = sampler2D(mesh_shader_params[draw_id].tf_texture_handle);
		out_colour = texture(tf_tx, vec2(sample_value_normalized, 1.0) ).rgb;

        if( radius > sample_value_normalized && radius < 0.96 ) discard;
    }

    float zero_value_radius = (- mesh_shader_params[draw_id].min_value) / (mesh_shader_params[draw_id].max_value - mesh_shader_params[draw_id].min_value);
    if(abs(radius - zero_value_radius) < 0.005) out_colour = vec3(1.0);
    if(radius > 0.96 && radius < 0.98) out_colour = vec3(0.0);

    float test = dFdx(radius);

    albedo_out = vec4(out_colour,1.0);
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;

    objID_out = mesh_shader_params[draw_id].probe_id;
    interactionData_out = vec4(0.0);
}
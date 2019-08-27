struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer LightParamsBuffer { LightParams light_params[]; };

uniform int light_cnt;

uniform sampler2D albedo_tx2D;
uniform sampler2D normal_tx2D;
uniform sampler2D depth_tx2D;

uniform mat4 inv_view_mx;
uniform mat4 inv_proj_mx;

in vec2 uv_coord;

out layout(location = 0) vec4 frag_out;

vec3 depthToWorldPos(float depth, vec2 uv) {
    float z = depth * 2.0 - 1.0;

    vec4 cs_pos = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 vs_pos = inv_proj_mx * cs_pos;

    // Perspective division
    vs_pos /= vs_pos.w;

    return vs_pos.xyz;

    vec4 ws_pos = inv_view_mx * vs_pos;

    return ws_pos.xyz;
}

vec3 lambert(vec3 normal, vec3 light_dir)
{
    return vec3( clamp(dot(normal,light_dir),0.0,1.0));
}

void main(void) {
    vec3 out_colour = vec3(0.0);

    vec3 colour = texture(albedo_tx2D,uv_coord).rgb;
    vec3 normal = texture(normal_tx2D,uv_coord).xyz;
    float depth = texture(depth_tx2D,uv_coord).r;

    if(depth < 0.001f){
        discard;
    }

    vec3 world_pos = depthToWorldPos(depth,uv_coord);

    for(int i=0; i<light_cnt; ++i)
    {
        vec3 light_dir = vec3(light_params[i].x,light_params[i].y,light_params[i].z) - world_pos;
        float d = length(light_dir); // centimeters to meters
        light_dir = normalize(light_dir);
        out_colour += colour * lambert(light_dir,normal) * light_params[i].intensity * (1.0/(d*d));
    }

    //  Temporary tone mapping
    out_colour = out_colour/(vec3(1.0)+out_colour);
    //	Temporary gamma correction
	out_colour = pow( out_colour, vec3(1.0/2.2) );

    frag_out = vec4(out_colour,1.0);
}
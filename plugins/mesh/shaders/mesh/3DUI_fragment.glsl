struct MeshShaderParams
{
    mat4 transform;
    vec4 color;
    int id;
    int highlighted;

    float padding0;
    float padding1;
};

struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer LightParamsBuffer { LightParams light_params[]; };

in vec3 worldPos;
in vec3 normal;
in vec3 colour;
in flat int id;

out layout(location = 0) vec4 frag_colour;
out layout(location = 1) int frag_id;

void main(void) {
    vec3 out_colour = vec3(0.0);

    for(int i=0; i<13; ++i)
    {
        vec3 light_dir = vec3(light_params[i].x,light_params[i].y,light_params[i].z) - worldPos;
        float d = length(light_dir) * 0.01; // centimeters to meters
        light_dir = normalize(light_dir);
        out_colour += colour * vec3( clamp(dot(normal,light_dir),0.0,1.0) * light_params[i].intensity * (1.0/(d*d)));
    }

    //  Temporary tone mapping
    out_colour = out_colour/(vec3(1.0)+out_colour);
    //	Temporary gamma correction
	out_colour = pow( out_colour, vec3(1.0/2.2) );

    frag_colour = vec4(out_colour,1.0);
    frag_id = id;
}
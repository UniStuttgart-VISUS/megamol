struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer LightParamsBuffer { LightParams[] light_params; };

in vec3 world_pos;
in vec3 normal;

out layout(location = 0) vec4 frag_colour;

void main(void) {
    vec3 colour = vec3(0.0);

    for(int i=0; i<13; ++i)
    {
        vec3 light_dir = vec3(light_params[i].x,light_params[i].y,light_params[i].z) - world_pos;
        float d = length(light_dir) * 0.01; // centimeters to meters
        light_dir = normalize(light_dir);
        colour += vec3( clamp(dot(normal,light_dir),0.0,1.0) * light_params[i].intensity * (1.0/(d*d)));
    }

    //  Temporary tone mapping
    colour = colour/(vec3(1.0)+colour);
    //	Temporary gamma correction
	colour = pow( colour, vec3(1.0/2.2) );

    frag_colour = vec4(colour,1.0);
}
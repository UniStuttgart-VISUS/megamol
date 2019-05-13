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

    for(int i=0; i<12; ++i)
    {
        vec3 light_dir = normalize(vec3(light_params[i].x,light_params[i].y,light_params[i].z) - world_pos);
        colour += vec3(dot(normal,light_dir) * light_params[i].intensity);
    }

    frag_colour = vec4(colour,1.0);
}
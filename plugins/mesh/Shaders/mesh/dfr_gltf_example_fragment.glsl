struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer LightParamsBuffer { LightParams light_params[]; };

in vec3 worldPos;
in vec3 normal;
in vec3 colour;

out layout(location = 0) vec3 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main(void) {
    albedo_out = colour;
    normal_out = normal;
    depth_out = gl_FragCoord.z;
}
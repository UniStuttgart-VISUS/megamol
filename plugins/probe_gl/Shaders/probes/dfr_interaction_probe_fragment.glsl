layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 colour;
layout(location = 3) flat in int objID;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;
layout(location = 3) out int objID_out;

void main(void) {
    albedo_out = vec4(colour,1.0);
    normal_out = normal;
    depth_out = gl_FragCoord.z;
    objID_out = objID;
}
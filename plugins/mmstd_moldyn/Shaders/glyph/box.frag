in vec4 objPos;
in vec4 camPos;
in vec4 lightPos;
in vec4 vertColor;

in vec3 invRad;

in mat3 rotMat;
in vec3 rotMatT0;
in vec3 rotMatT1; // rotation matrix from the quaternion
in vec3 rotMatT2;
in mat3 rotMatIT;

in vec3 normal;

layout (location = 0) out vec4 out_frag_color;

void main() {
    //vec3 normal = vec3(1,0,0);
    //normal = rotMatIT * normal;
    out_frag_color = vec4(LocalLighting(ray, normal, lightPos.xyz, vertColor), 1.0);
}
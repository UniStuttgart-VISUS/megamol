#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_EXT_gpu_shader4 : require

uniform int uOuter0 = 1;
uniform int uOuter1 = 16;

uniform int instanceOffset;

out vec4 myColor[];

layout( vertices = 4 ) out;

struct CAType {
    float x, y, z;
    int type;
};

layout(std430, binding = 2) buffer shader_data {
    CAType cAlphas[];
};

void main() {
    CAType ca = cAlphas[gl_PrimitiveID + gl_InvocationID + instanceOffset];
    myColor[gl_InvocationID] = vec4(0, 0, 0, 1);
    if (ca.type == 1) {
        myColor[gl_InvocationID] = vec4(0, 0, 1, 1);
    } else if (ca.type == 2) {
        myColor[gl_InvocationID] = vec4(1, 0, 0, 1);
    } else if (ca.type == 3) {
        myColor[gl_InvocationID] = vec4(0, 1, 0, 1);
    }
    gl_out[gl_InvocationID].gl_Position = vec4(ca.x, ca.y, ca.z, 1.0f);
    gl_TessLevelOuter[0] = float( uOuter0);
    gl_TessLevelOuter[1] = float( uOuter1);
}

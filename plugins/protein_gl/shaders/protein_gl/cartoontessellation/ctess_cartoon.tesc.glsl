#version 430

uniform int uOuter0 = 16;
uniform int uOuter1 = 16;

uniform int minInner = 6;
uniform int maxInner = 30;
uniform int minOuter = 6;
uniform int maxOuter = 30;

uniform int instanceOffset = 0;

layout(vertices = 4) out;

out int id[];

struct CAlpha {
    vec4 pos;
    vec3 dir;
    int type;
};

layout(std430, binding = 2) buffer shader_data {
    CAlpha atoms[];
};

void main() {
        
    int atomPos = gl_PrimitiveID + (gl_InvocationID % 2);
    gl_out[gl_InvocationID].gl_Position = atoms[atomPos].pos;
    id[gl_InvocationID] = atomPos;
    
    if(gl_InvocationID == 0)
    {
        gl_TessLevelOuter[0] = float(uOuter0);
        gl_TessLevelOuter[1] = float(uOuter1);
        gl_TessLevelOuter[2] = float(uOuter0);
        gl_TessLevelOuter[3] = float(uOuter1);
        
        gl_TessLevelInner[0] = float(uOuter0);
        gl_TessLevelInner[1] = float(uOuter1);
    }
}

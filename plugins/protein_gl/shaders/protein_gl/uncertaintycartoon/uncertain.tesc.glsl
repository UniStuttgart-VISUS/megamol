#version 430
///////////////////////////
// TESSELATION - control //
///////////////////////////

/////////////
// DEFINES //
/////////////
#define STRUCT_COUNT 9   // must be equal to 'UncertaintyDataCall::secStructure::NOE'

////////////
// LAYOUT //
////////////  
layout(vertices = 4) out;
////////
// IN //
////////
in vec4 camPos[];

/////////
// OUT //
/////////    
out int id[];
///////////////
// variables //
///////////////
uniform int uOuter0        = 16;
uniform int uOuter1        = 16;
// uniform int minInner    = 6;                                     // UNUSED - just as INFO
// uniform int maxInner    = 30;                                    // UNUSED - just as INFO - why 30?
// uniform int minOuter    = 6;                                     // UNUSED - just as INFO
// uniform int maxOuter    = 30;                                    // UNUSED - just as INFO - why 30?
uniform int instanceOffset = 0;                                     // UNUSED ?

uniform int tessLevel;
struct CAlpha
{
    vec4  pos;
    vec3  dir;
    int   colIdx;        
    vec4  col;
    float uncertainty;
    int   flag; 
    float unc[STRUCT_COUNT];
    int   sortedStruct[STRUCT_COUNT];
};
layout(std430, binding = 2) buffer shader_data {
    CAlpha atoms[];
};
//////////
// MAIN //
//////////
void main() {
    
    int atomPos  = gl_PrimitiveID + (gl_InvocationID % 2);          // 4 invocations because of "layout(vertices = 4) out".
                                                                    // output vertices with index 0 and 2 belong to atomPos with inv%2=0 and 
                                                                    // vertices with index 1 and 3 belong to atomPos with inv%2=1 (???)
    gl_out[gl_InvocationID].gl_Position = atoms[atomPos].pos; 
    id[gl_InvocationID]                 = atomPos;

    vec4 cp = camPos[0];                                            // UNUSED ? 

    if(gl_InvocationID == 0)
    {
        int tessOuter0 = tessLevel; // uOuter0
        int tessOuter1 = tessLevel; // uOuter1
        
        gl_TessLevelOuter[0] = float(tessOuter0);
        gl_TessLevelOuter[1] = float(tessOuter1);
        gl_TessLevelOuter[2] = float(tessOuter0);
        gl_TessLevelOuter[3] = float(tessOuter1);
    
        gl_TessLevelInner[0] = float(tessOuter0);
        gl_TessLevelInner[1] = float(tessOuter1);
    }
}

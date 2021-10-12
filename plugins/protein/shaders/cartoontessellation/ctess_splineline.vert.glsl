#version 430

#include "simplemolecule/sm_common_defines.glsl"

uniform vec4 viewAttr;

uniform float scaling;

uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;
uniform int instanceOffset;

uniform mat4 MVinv;
uniform mat4 MVP;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;

uniform vec4 inConsts1;
uniform sampler1D colTab;

out vec4 objPos;
out vec4 camPos;
out float squarRad;
out float rad;
out vec4 vertColor;

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w

void main(void) {
    float theColIdx;
    vec4 theColor;
    vec4 inPos;

    vertColor = theColor;
        
    rad *= scaling;
    squarRad = rad * rad;

    // object pivot point in object space    
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)

    // calculate cam position
    camPos = MVinv[3]; // (C) by Christoph
    camPos.xyz -= objPos.xyz; // cam pos to glyph space
    
    gl_Position = objPos;
    gl_PointSize = 2.0;
}
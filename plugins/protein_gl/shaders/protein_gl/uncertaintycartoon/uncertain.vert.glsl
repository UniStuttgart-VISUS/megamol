#version 430

#define CLIP
#define DEPTH
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

//#define RETICLE
//#define DISCARD_COLOR_MARKER
#ifdef DEBUG
#undef CLIP
#define RETICLE
#define AXISHINTS
#endif // DEBUG

//#define BULLSHIT

#ifndef FLACH
#define FLACH
#endif

/////////////////////////////
// VERTEX - common defines //
/////////////////////////////

/////////////
// DEFINES //
/////////////
#define CONSTRAD    inConsts1.x
#define MIN_COLV    inConsts1.y
#define MAX_COLV    inConsts1.z
#define COLTAB_SIZE inConsts1.w

/////////
// OUT //
/////////
out vec4  objPos;
out vec4  camPos;
//out vec4  lightPos;
out float squarRad;                                                 // radius squared
out float rad;                                                      // radius
out vec4  vertColor;

/////////////////////
// INPUT variables //
/////////////////////   
uniform vec4      viewAttr;                             

uniform float     scaling;                                          // UNUSED - only used by slpine shader    

uniform vec3      camIn;
uniform vec3      camUp;
uniform vec3      camRight;

// clipping plane attributes
uniform vec4      clipDat;
uniform vec4      clipCol;

uniform mat4      MVinv;
uniform mat4      MVinvtrans;
uniform mat4      MVP;
uniform mat4      MVPinv;
uniform mat4      MVPtransp;

uniform vec4      inConsts1;

uniform int       instanceOffset;                                   // UNUSED - for what ? - always 0 assigned
// uniform sampler1D colTab;                                        // UNUSED - 1D texture

////////////////////////////
// VERTEX - main - params //
////////////////////////////

//////////
// MAIN //
//////////
void main(void) {
    float theColIdx;                                                // UNUSED - NOT initialised ? (= ... ?)
    vec4  theColor;                                                 // UNUSED - NOT initialised ? (= gl_Color)
    vec4  inPos;                                                    // UNUSED - NOT initialised ? (= gl_Position)
    //////////////////////////
    // VERTEX - main - rest //
    //////////////////////////

    vertColor = theColor;
        
    rad      *= scaling;
    squarRad  = rad * rad;

    //////////////////////////////////////////
    // VERTEX - main - position translation //
    //////////////////////////////////////////

    // object pivot point in object space    
    objPos       = inPos;                                               // no w-div needed, because w is 1.0 (Because I know)

    // calculate cam position
    camPos       = MVinv[3];                                            // (C) by Christoph
    camPos.xyz  -= objPos.xyz;                                          // cam pos to glyph space

    // calculate light position in glyph space
    //lightPos     = MVinv * gl_LightSource[0].position;
    
    gl_Position  = objPos;
    gl_PointSize = 2.0;
}

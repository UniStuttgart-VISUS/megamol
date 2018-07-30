uniform vec4 viewAttr;

uniform float scaling;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;

uniform vec4 inConsts1;
attribute float colIdx;
uniform sampler1D colTab;

varying vec4 objPos;
varying vec4 camPos;
varying vec4 lightPos;
varying float squarRad;
varying float rad;

#ifdef DEFERRED_SHADING
varying float pointSize;
#endif

#ifdef RETICLE
varying vec2 centerFragment;
#endif // RETICLE

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w
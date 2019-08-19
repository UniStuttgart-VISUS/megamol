#extension GL_ARB_explicit_attrib_location : require   // glsl version 130
#extension GL_ARB_conservative_depth       : require   // glsl version 130
layout (depth_greater) out float gl_FragDepth; 

#ifdef BACKSIDE_ENABLED
uniform float hitsideFlag;
#endif // BACKSIDE_ENABLED

//#define DISCARD_COLOR_MARKER

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;
uniform float alphaScaling;

uniform vec4 viewAttr;

FLACH in vec4 objPos;
FLACH in vec4 camPos;
//FLACH in vec4 lightPos;
FLACH in float squarRad;
FLACH in float rad;
FLACH in float effectiveDiameter;

#ifdef RETICLE
FLACH in vec2 centerFragment;
#endif // RETICLE

in vec4 vertColor;
layout(location = 0) out vec4 outColor;

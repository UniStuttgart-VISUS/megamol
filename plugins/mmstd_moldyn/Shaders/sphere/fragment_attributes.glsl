
#extension GL_ARB_explicit_attrib_location : require   // glsl version 130
#extension GL_ARB_conservative_depth       : require   // glsl version 130
layout (depth_greater) out float gl_FragDepth; 

FLACH in vec4 objPos;
FLACH in vec4 camPos;
FLACH in vec4 outlightDir;
FLACH in float squarRad;
FLACH in float rad;
FLACH in vec4 vertColor;

#ifdef RETICLE
FLACH in vec2 centerFragment;
#endif // RETICLE

#ifdef BACKSIDE_ENABLED
uniform float hitsideFlag;
#endif // BACKSIDE_ENABLED

//#define DISCARD_COLOR_MARKER
//#undef CLIP
//#undef DEPTH

uniform vec4 clipDat;
uniform vec4 clipCol;

uniform vec4 viewAttr;

uniform mat4 MVPinv;
uniform mat4 MVPtransp;

// Only used by SPLAT render mode:
uniform float alphaScaling;
FLACH in float effectiveDiameter;

// Only used by AMBIENT OCLUSION render mode:
uniform bool inUseHighPrecision;
out vec4 outNormal;

// Only used by OUTLINE render mode:
uniform float outlineSize;

layout(location = 0) out vec4 outColor;

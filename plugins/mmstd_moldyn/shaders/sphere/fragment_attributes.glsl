
FLACH in vec4 objPos;
FLACH in vec4 camPos;
FLACH in vec4 outlightDir;
FLACH in float squarRad;
FLACH in float rad;
FLACH in vec4 vertColor;

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

// SPLAT
uniform float alphaScaling;
FLACH in float effectiveDiameter;

// AMBIENT OCLUSION
uniform bool inUseHighPrecision;
out vec4 outNormal;

// OUTLINE / ifdef RETICLE
uniform float outlineWidth;
FLACH in float fragmentRadius;
FLACH in vec2 centerFragment;

layout(location = 0) out vec4 outColor;


#ifdef BACKSIDE_ENABLED
uniform float hitsideFlag;
#endif // BACKSIDE_ENABLED

uniform vec4 clipDat;
uniform vec4 clipCol;

uniform vec4 viewAttr;

uniform mat4 MVPinv;
uniform mat4 MVPtransp;

// SPLAT
uniform float alphaScaling;

// AMBIENT OCLUSION
uniform bool inUseHighPrecision;

// OUTLINE
uniform float outlineWidth = 0.0;


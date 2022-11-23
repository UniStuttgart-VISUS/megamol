#ifdef BACKSIDE_ENABLED
uniform float hitsideFlag;
#endif // BACKSIDE_ENABLED

// clipping plane attributes
uniform vec4 clipDat;
uniform vec3 clipCol;

uniform vec3 posOrigin;
uniform vec3 posExtents;
uniform vec3 aoSampDist;
uniform float aoSampFact;

uniform vec4 viewAttr;
uniform vec2 frustumPlanes;

FLACH varying vec4 objPos;
FLACH varying vec4 camPos;
FLACH varying vec4 lightPos;
FLACH varying vec2 radii; // vec2(r, r^2)

#ifdef RETICLE
FLACH varying vec2 centerFragment;
#endif // RETICLE

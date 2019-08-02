#extension GL_ARB_explicit_attrib_location : require   // glsl version 130
#extension GL_ARB_conservative_depth       : require   // glsl version 130
layout (depth_greater) out float gl_FragDepth; 


#ifdef BACKSIDE_ENABLED
uniform float hitsideFlag;
#endif // BACKSIDE_ENABLED
//#define DISCARD_COLOR_MARKER
//#undef CLIP
//#undef DEPTH

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;

uniform vec4 viewAttr;

uniform mat4 MVPinv;
uniform mat4 MVPtransp;

FLACH in vec4 objPos;
FLACH in vec4 camPos;
FLACH in vec4 lightPos;
FLACH in float squarRad;
FLACH in float rad;
FLACH in vec4 vertColor;

#ifdef RETICLE
FLACH in vec2 centerFragment;
#endif // RETICLE

layout(location = 0) out vec4 outColor;


// Forward declaration of lighting function.
vec3 LocalLighting(const in vec3 ray, const in vec3 normal, const in vec3 lightPos, const in vec3 color);


void main(void) {

    vec4 coord;
    vec3 ray;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    

    // transform fragment coordinates from view coordinates to object coordinates.
    //coord = MVPinv * coord;
    coord = MVPinv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space
    

    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);

    // chose color for lighting
    vec4 color = vertColor;
    //vec4 color = vec4(uplParams.xyz, 1.0);

    // calculate the geometry-ray-intersection
    float d1 = -dot(camPos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(camPos.xyz, camPos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = squarRad - d2s;                        // square of difference of projected length and lambda
#ifdef CLIP
    if (radicand < 0.0) { 
#ifdef DISCARD_COLOR_MARKER
        color = vec4(1.0, 0.0, 0.0, 1.0);       
#else // DISCARD_COLOR_MARKER
        discard; 
#endif // DISCARD_COLOR_MARKER
    }
#endif // CLIP

    float sqrtRadicand = sqrt(radicand);
#ifdef BACKSIDE_ENABLED
    lambda = d1 - sqrtRadicand * hitsideFlag;             // lambda
#else // BACKSIDE_ENABLED
    lambda = d1 - sqrtRadicand;                           // lambda
#endif // BACKSIDE_ENABLED

    vec3 sphereintersection = lambda * ray + camPos.xyz;    // intersection point
    vec3 normal = sphereintersection / rad;


    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        vec3 planeNormal = normalize(clipDat.xyz);
        vec3 clipPlaneBase = planeNormal * clipDat.w;
        float d = -dot(planeNormal, clipPlaneBase - objPos.xyz);
        float dist1 = dot(sphereintersection, planeNormal) + d;
        float dist2 = d;
        float t = -(dot(planeNormal, camPos.xyz) + d) / dot(planeNormal, ray);
        vec3 planeintersect = camPos.xyz + t * ray;
        if (dist1 > 0.0) {
            if (dist2 < rad) {
                if (length(planeintersect) < rad) {
                    sphereintersection = planeintersect;
                    normal = planeNormal;
                    color = mix(color, vec4(clipCol.rgb, 1.0), clipCol.a);
                } else {
                    discard;
                }
            } else {
                discard;
            }
        }
    }


    // "calc" normal at intersection point
#ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, lightPos.w);
#endif // SMALL_SPRITE_LIGHTING

#ifdef AXISHINTS
    // debug-axis-hints
    float mc = min(abs(normal.x), min(abs(normal.y), abs(normal.z)));
    if (mc < 0.05)            { color = vec3(0.5); }
    if (abs(normal.x) > 0.98) { color = vec3(1.0, 0.0, 0.0); }
    if (abs(normal.y) > 0.98) { color = vec3(0.0, 1.0, 0.0); }
    if (abs(normal.z) > 0.98) { color = vec3(0.0, 0.0, 1.0); }
    if (normal.x < -0.99)     { color = vec3(0.5); }
    if (normal.y < -0.99)     { color = vec3(0.5); }
    if (normal.z < -0.99)     { color = vec3(0.5); }
#endif // AXISHINTS

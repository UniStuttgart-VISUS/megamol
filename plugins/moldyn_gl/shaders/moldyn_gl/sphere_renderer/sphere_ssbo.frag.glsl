#version 330

//////// #include "moldyn_gl/sphere_renderer/inc/fragment_extensions.inc.glsl"  -------> keep but modify
#extension GL_ARB_explicit_attrib_location : enable    // glsl version 130
#ifdef __AMD__
#extension GL_AMD_conservative_depth       : require   // glsl version 130
#else
#extension GL_ARB_conservative_depth       : require   // glsl version 130
#endif
layout (depth_greater) out float gl_FragDepth;
////////

//////// #include "commondefines.glsl" ------> remove from spherer renderer?
#define CLIP
#define DEPTH
#define WITH_SCALING
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

//#define HALO
#ifdef HALO
    #define HALO_RAD 3.0
#endif // HALO

//#define DEBUG
#ifdef DEBUG
    #undef CLIP
    #define RETICLE
    #define AXISHINTS
#endif // DEBUG

#ifndef FLACH
    #define FLACH
#endif // FLACH
////////

////////#include "moldyn_gl/sphere_renderer/inc/fragment_attributes.inc.glsl" ------> split
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
FLACH in float sphere_frag_radius;
FLACH in vec2 sphere_frag_center;
// OUTLINE
uniform float outlineWidth = 0.0;


layout(location = 0) out vec4 outColor;
////////

//////// #include "lightdirectional.glsl" -----> keep
// DIRECTIONAL LIGHTING (Blinn Phong)

// ray:      the eye to fragment ray vector
// normal:   the normal of this fragment
// lightdir: the direction of the light 
// color:    the base material color

//#define USE_SPECULAR_COMPONENT

vec3 LocalLighting(const in vec3 ray, const in vec3 normal, const in vec3 lightdir, const in vec3 color) {

    vec3 lightdirn = normalize(-lightdir); // (negativ light dir for directional lighting)

    vec4 lightparams = vec4(0.2, 0.8, 0.4, 10.0);
#define LIGHT_AMBIENT  lightparams.x
#define LIGHT_DIFFUSE  lightparams.y
#define LIGHT_SPECULAR lightparams.z
#define LIGHT_EXPONENT lightparams.w

    float nDOTl = dot(normal, lightdirn);

    vec3 specular_color = vec3(0.0, 0.0, 0.0);
#ifdef USE_SPECULAR_COMPONENT
    vec3 r = normalize(2.0 * vec3(nDOTl) * normal - lightdirn);
    specular_color = LIGHT_SPECULAR * vec3(pow(max(dot(r, -ray), 0.0), LIGHT_EXPONENT));
#endif // USE_SPECULAR_COMPONENT

    return LIGHT_AMBIENT  * color 
         + LIGHT_DIFFUSE  * color * max(nDOTl, 0.0)
         + specular_color;
}
/////////

vec3 computeRay(vec4 fragCoord, vec4 viewAttr, mat4 MVPinv, vec4 camPos, vec4 objPos){
    // transform fragment coordinates from window coordinates to view coordinates.
    vec4 coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    
    // transform fragment coordinates from view coordinates to object coordinates.
    //coord = MVPinv * coord;
    coord = MVPinv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space

    // calc the viewing ray
    return normalize(coord.xyz - camPos.xyz);
};

struct Intersection{
    vec3 position;
    vec3 normal;
    vec4 color;
};

Intersection computeRaySphereIntersection(vec3 ray, vec4 camPos, vec4 objPos, vec4 color, float squarRad, vec4 clipDat, vec4 clipCol){
    Intersection retval;
    retval.color = color;

    // calculate the geometry-ray-intersection
    float b = -dot(camPos.xyz, ray);           // projected length of the cam-sphere-vector onto the ray
    vec3 temp = camPos.xyz + b*ray;
    float delta = squarRad - dot(temp, temp);  // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)

#ifdef CLIP
    if (delta < 0.0) {
#ifdef DISCARD_COLOR_MARKER
        retval.color = vec4(1.0, 0.0, 0.0, 1.0);       
#else // DISCARD_COLOR_MARKER
        discard; 
#endif // DISCARD_COLOR_MARKER
    }
#endif // CLIP

    float c = dot(camPos.xyz, camPos.xyz)-squarRad;

    float s = b < 0.0f ? -1.0f : 1.0f;
    float q = b + s*sqrt(delta);
    float lambda = min(c/q, q);

    retval.position = lambda * ray + camPos.xyz;    // intersection point
    retval.normal = retval.position / rad;

    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        vec3 planeNormal = normalize(clipDat.xyz);
        vec3 clipPlaneBase = planeNormal * clipDat.w;
        float d = -dot(planeNormal, clipPlaneBase - objPos.xyz);
        float dist1 = dot(retval.position, planeNormal) + d;
        float dist2 = d;
        float t = -(dot(planeNormal, camPos.xyz) + d) / dot(planeNormal, ray);
        vec3 planeintersect = camPos.xyz + t * ray;
        if (dist1 > 0.0) {
            if (dist2 < rad) {
                if (length(planeintersect) < rad) {
                    retval.position = planeintersect;
                    retval.normal = planeNormal;
                    retval.color = mix(retval.color, vec4(clipCol.rgb, 1.0), clipCol.a);
                } else {
                    discard;
                }
            } else {
                discard;
            }
        }
    }

    return retval;
};

vec4 axisHintsColor(vec3 normal){
    // debug-axis-hints
    vec4 retval = vec4(0.0);
    float mc = min(abs(normal.x), min(abs(normal.y), abs(normal.z)));
    if (mc < 0.05) {
        retval = vec4(0.5, 0.5, 0.5, 1.0);
    }
    if (abs(normal.x) > 0.98) {
        retval = vec4(1.0, 0.0, 0.0, 1.0);
    }
    if (abs(normal.y) > 0.98) {
        retval = vec4(0.0, 1.0, 0.0, 1.0);
    }
    if (abs(normal.z) > 0.98) {
        retval = vec4(0.0, 0.0, 1.0, 1.0);
    }
    if (normal.x < -0.99) {
        retval = vec4(0.5, 0.5, 0.5, 1.0);
    }
    if (normal.y < -0.99) {
        retval = vec4(0.5, 0.5, 0.5, 1.0);
    }
    if (normal.z < -0.99) {
        retval = vec4(0.5, 0.5, 0.5, 1.0);
    }
    return retval;
};

float computeDepthValue(vec3 position){
    float depth = dot(MVPtransp[2], vec4(position, 1.0));
    float depthW = dot(MVPtransp[3], vec4(position, 1.0));
    return ((depth / depthW) + 1.0) * 0.5;
};

vec4 reticleColor(vec4 color, vec4 fragCoord, vec2 sphere_frag_center){
    vec4 retval = color;
    if (min(abs(fragCoord.x - sphere_frag_center.x), abs(fragCoord.y - sphere_frag_center.y)) < 2.0f) {
        //outColor.rgb = vec3(1.0, 1.0, 0.5);
        retval.rgb += vec3(0.3, 0.3, 0.5);
    }
    return retval;
};

//////// #include "moldyn_gl/sphere_renderer/inc/fragment_mainstart.inc.glsl" ------> get rid of
void main(void) {

    vec3 ray = computeRay(gl_FragCoord,viewAttr,MVPinv,camPos,objPos);

    Intersection sphereIntersection = computeRaySphereIntersection(ray,camPos,objPos,vertColor,squarRad,clipDat,clipCol);

    // "calc" normal at intersection point
#ifdef SMALL_SPRITE_LIGHTING
    sphereIntersection.normal = mix(-ray, sphereIntersection.normal, outlightDir.w);
#endif // SMALL_SPRITE_LIGHTING

#ifdef AXISHINTS
    sphereIntersection.color = axisHintsColor(normal);
#endif // AXISHINTS

//////// #include "moldyn_gl/sphere_renderer/inc/fragment_out-lighting.inc.glsl" ----> get rid of
    // Phong lighting with directional light
    outColor = vec4(LocalLighting(ray, sphereIntersection.normal, outlightDir.xyz, sphereIntersection.color.rgb), sphereIntersection.color.a);
////////

//////// #include "moldyn_gl/sphere_renderer/inc/fragment_out-depth.inc.glsl" ----> replace and get rif of
// Calculate depth
#ifdef DEPTH

    float depth = computeDepthValue(sphereIntersection.position + objPos.xyz);
    gl_FragDepth = depth;

#ifndef CLIP
    gl_FragDepth = (delta < 0.0) ? 1.0 : depth;
    outColor.rgb = (delta < 0.0) ? vertColor.rgb : outColor.rgb;
#endif // CLIP

#ifdef DISCARD_COLOR_MARKER
    gl_FragDepth = computeDepthValue(objPos.xyz);
#endif // DISCARD_COLOR_MARKER

#endif // DEPTH
////////

//////// #include "moldyn_gl/sphere_renderer/inc/fragment_mainend.inc.glsl" -----> replace and get rid of
#ifdef RETICLE
    outColor.rgb reticleColor
#endif // RETICLE

}
////////

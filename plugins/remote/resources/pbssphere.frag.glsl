#version 140

#define CLIP
#define DEPTH
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

#ifdef DEBUG
#undef CLIP
#define RETICLE
#define AXISHINTS
#endif // DEBUG

//#define BULLSHIT

#ifndef FLACH
#define FLACH
#endif

// TODO: Implementation is wrong! Does positional Lighting instead of directional lighting!

// ray:      the eye to fragment ray vector
// normal:   the normal of this fragment
// lightPos: the position of the light source
// color:    the base material color
vec3 LocalLighting(const in vec3 ray, const in vec3 normal, const in vec3 lightPos, const in vec3 color) {
    // TODO: rewrite!
    vec3 lightDir = normalize(lightPos);

    vec4 lightparams = vec4(0.2, 0.8, 0.4, 10.0);
#define LIGHT_AMBIENT lightparams.x
#define LIGHT_DIFFUSE lightparams.y
#define LIGHT_SPECULAR lightparams.z
#define LIGHT_EXPONENT lightparams.w
    float nDOTl = dot(normal, lightDir);

    vec3 r = normalize(2.0 * vec3(nDOTl) * normal - lightDir);
    return LIGHT_AMBIENT * color 
        + LIGHT_DIFFUSE * color * max(nDOTl, 0.0) 
        + LIGHT_SPECULAR * vec3(pow(max(dot(r, -ray), 0.0), LIGHT_EXPONENT));
}

#extension GL_ARB_conservative_depth:require
layout (depth_greater) out float gl_FragDepth; // we think this is right
// this should be wrong //layout (depth_less) out float gl_FragDepth;
#extension GL_ARB_explicit_attrib_location : enable

in mat4 MVP;
in mat4 MVPinv;
in mat4 MVPtransp;

// uniform mat4 MVP;
// uniform mat4 MVPinv;
// uniform mat4 MVPtransp;

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

FLACH in vec4 objPos;
FLACH in vec4 camPos;
FLACH in vec4 lightPos;
FLACH in float squarRad;
FLACH in float rad;
FLACH in vec4 vertColor;

out layout(location = 0) vec4 outCol;

#ifdef RETICLE
FLACH in vec2 centerFragment;
#endif // RETICLE

void main(void) {
    vec4 coord;
    vec3 ray;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    

    // transform fragment coordinates from view coordinates to object coordinates.
    //coord = gl_ModelViewProjectionMatrixInverse * coord;
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

// phong lighting with directional light
    outCol = vec4(LocalLighting(ray, normal, lightPos.xyz, color.rgb), color.a);
    //outCol = color;

    // calculate depth
#ifdef DEPTH
    vec4 Ding = vec4(sphereintersection + objPos.xyz, 1.0);
    float depth = dot(MVPtransp[2], Ding);
    float depthW = dot(MVPtransp[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
#ifndef CLIP
    gl_FragDepth = (radicand < 0.0) ? 1.0 : ((depth / depthW) + 1.0) * 0.5;
    outCol.rgb = (radicand < 0.0) ? vertColor.rgb : outCol.rgb;
#endif // CLIP

#ifdef DISCARD_COLOR_MARKER
    Ding = vec4(objPos.xyz, 1.0);
    depth = dot(MVPtransp[2], Ding);
    depthW = dot(MVPtransp[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
#endif // DISCARD_COLOR_MARKER

#endif // DEPTH

#ifdef RETICLE
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    if (min(abs(coord.x - centerFragment.x), abs(coord.y - centerFragment.y)) < 0.002) {
        //outCol.rgb = vec3(1.0, 1.0, 0.5);
        outCol.rgb += vec3(0.3, 0.3, 0.5);
    }
#endif // RETICLE
//    outCol.rgb = normal;
//outCol = vec4(1.0, 0.0, 0.0, 1.0);
}
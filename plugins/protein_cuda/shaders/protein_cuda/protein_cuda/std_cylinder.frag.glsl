#version 110

#include "protein_cuda/protein_cuda/commondefines.glsl"
#include "lightdirectional.glsl"

//#undef CLIP

// #define RENDER_CAPS

uniform vec4 viewAttr;

FLACH varying vec4 objPos;
FLACH varying vec4 camPos;
FLACH varying vec4 lightPos;
FLACH varying vec3 radz;

FLACH varying vec3 rotMatT0;
FLACH varying vec3 rotMatT1; // rotation matrix from the quaternion
FLACH varying vec3 rotMatT2;

#ifdef RETICLE
FLACH varying vec2 centerFragment;
#endif // RETICLE

void main(void) {
    vec4 coord;
    vec3 ray, tmp;
    const float maxLambda = 50000.0;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);


    // transform fragment coordinates from view coordinates to object coordinates.
    coord = gl_ModelViewProjectionMatrixInverse * coord;
    coord /= coord.w;
    coord -= objPos; // ... and move


    // calc the viewing ray
    ray = rotMatT0 * coord.x + rotMatT1 * coord.y + rotMatT2 * coord.z;
    ray = normalize(ray - camPos.xyz);


    // calculate the geometry-ray-intersection

    // cylinder parameters
#define CYL_RAD radz.x
#define CYL_RAD_SQ radz.y
#define CYL_HALF_LEN radz.z

    float rDc = dot(ray.yz, camPos.yz);
    float rDr = dot(ray.yz, ray.yz);

    float radicand = (rDc * rDc) - (rDr * (dot(camPos.yz, camPos.yz) - CYL_RAD_SQ));

#ifdef CLIP
    if (radicand < 0.0) { discard; }
#endif // CLIP

   float radix = sqrt(radicand);
    vec2 lambdas = vec2((-rDc - radix) / rDr, (-rDc + radix) / rDr);

    // calculations for cylinder caps
    vec3 cylPt1 = camPos.xyz + ray * lambdas.x;
    vec3 cylPt2 = camPos.xyz + ray * lambdas.y;

#ifdef RENDER_CAPS
    bool cylPt1Right = (cylPt1.x > 0.0);
#endif // RENDER_CAPS
    bool cylPt1Valid = (cylPt1.x <= CYL_HALF_LEN) && (cylPt1.x >= -CYL_HALF_LEN); // trim cylinder
#ifdef RENDER_CAPS
    bool cylPt2Valid = cylPt1Right ? (cylPt2.x <= CYL_HALF_LEN) : (cylPt2.x >= -CYL_HALF_LEN);
#else
    bool cylPt2Valid = (cylPt2.x <= CYL_HALF_LEN) && (cylPt2.x >= -CYL_HALF_LEN);
#endif // RENDER_CAPS

#ifdef RENDER_CAPS
    lambdas.x = cylPt1Valid ? lambdas.x : (((cylPt1Right ? CYL_HALF_LEN : -CYL_HALF_LEN) - camPos.x) / ray.x);
    lambdas.x = (cylPt1Valid || cylPt2Valid) ? lambdas.x : maxLambda;

#ifdef CLIP
    if (lambdas.x >= maxLambda) { discard; }
#else // CLIP
    if (lambdas.x >= maxLambda) { radicand = -1.0; }
#endif // CLIP

#else // RENDER_CAPS
    lambdas.x = (cylPt1Valid) ? lambdas.x : lambdas.y;

#ifdef CLIP
    if ((!cylPt1Valid && !cylPt2Valid)) { discard; }
#else // CLIP
    if ((!cylPt1Valid && !cylPt2Valid)) { radicand = -1.0; }
#endif // CLIP
#endif // RENDER_CAPS

    vec3 intersection = camPos.xyz + ray * lambdas.x;
#ifdef RENDER_CAPS
    vec3 normal = (cylPt1Valid ? vec3(0.0, normalize(intersection.yz)) : (cylPt1Right ? vec3(1.0, 0.0, 0.0) : vec3(-1.0, 0.0, 0.0)));
#else // RENDER_CAPS
    vec3 normal = vec3(0.0, normalize(intersection.yz));
#endif // RENDER_CAPS


    // chose color for lighting
    vec3 color = gl_Color.rgb;

    // set fragment color (kroneml)
    if( cylPt1.x > 0.0 )
    {
        color = gl_SecondaryColor.rgb;
    }
    else
    {
        color = gl_Color.rgb;
    }

    // phong lighting with directional light
    gl_FragColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);


    // calculate depth
#ifdef DEPTH
    tmp = intersection;
    intersection.x = dot(rotMatT0, tmp.xyz);
    intersection.y = dot(rotMatT1, tmp.xyz);
    intersection.z = dot(rotMatT2, tmp.xyz);

    intersection += objPos.xyz;

    vec4 Ding = vec4(intersection, 1.0);
    float depth = dot(gl_ModelViewProjectionMatrixTranspose[2], Ding);
    float depthW = dot(gl_ModelViewProjectionMatrixTranspose[3], Ding);
#ifndef CLIP
    if (radicand < 0.0) {
        gl_FragColor = gl_SecondaryColor;
        depth = 1.0;
        depthW = 1.0;
    }
#endif // CLIP
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
#endif // DEPTH

#ifdef RETICLE
    coord = gl_FragCoord
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);
    if (min(abs(coord.x - centerFragment.x), abs(coord.y - centerFragment.y)) < 0.002) {
        gl_FragColor.rgb += vec3(0.3, 0.3, 0.5);
    }
#endif // RETICLE
//    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
//    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}

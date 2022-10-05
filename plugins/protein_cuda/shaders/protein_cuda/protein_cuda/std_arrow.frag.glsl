#version 140

#include "protein_cuda/protein_cuda/commondefines.glsl"
#include "lightdirectional.glsl"

// #undef CLIP
#extension GL_ARB_gpu_shader5 : enable
#extension GL_EXT_geometry_shader4 : enable

#define RENDER_CAPS

uniform vec4 viewAttr;
uniform mat4 modelview;
uniform mat4 proj;

in vec4 objPos;
in vec4 camPos;
in vec4 light;
in vec4 radz; /* (cyl-Rad, tip-Rad, overall-Len, tip-Len) */
in vec3 rotMatT0;
in vec3 rotMatT1; // rotation matrix from the quaternion
in vec3 rotMatT2;
in vec4 colOut;

#ifdef RETICLE
in vec2 centerFragment;
#endif // RETICLE

mat4 modelviewproj = proj*modelview; // TODO Move this to the CPU?
mat4 modelviewProjInv = inverse(modelviewproj);
mat4 modelviewProjTrans = transpose(modelviewproj);

void main(void) {
    vec4 coord;
    vec3 ray, tmp;
    const float maxLambda = 50000.0;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform fragment coordinates from view coordinates to object coordinates.
    coord = modelviewProjInv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and move


    // calc the viewing ray
    ray = rotMatT0 * coord.x + rotMatT1 * coord.y + rotMatT2 * coord.z;
    ray = normalize(ray - camPos.xyz);


    // calculate the geometry-ray-intersection

    // cylinder parameters
#define CYL_RAD radz.x
#define CYL_RAD_SQ radz.x * radz.x
#define CYL_LEN radz.z
#define TIP_RAD radz.y
#define TIP_LEN radz.w

    // super unoptimized cone code

    float coneF = TIP_RAD / TIP_LEN;
    coneF *= coneF;
    float coneA = coneF * ray.x * ray.x - ray.y * ray.y - ray.z * ray.z;
    float coneB = 2.0 * (coneF * ray.x * camPos.x - ray.y * camPos.y - ray.z * camPos.z);
    float coneC = coneF * camPos.x * camPos.x - camPos.y * camPos.y - camPos.z * camPos.z;

    float rDc = dot(ray.yz, camPos.yz);
    float rDr = dot(ray.yz, ray.yz);

    vec2 radicand = vec2(
        (rDc * rDc) - (rDr * (dot(camPos.yz, camPos.yz) - CYL_RAD_SQ)),
        coneB * coneB - 4.0 * coneA * coneC);
    vec2 divisor = vec2(rDr, 2.0 * coneA);
    vec2 radix = sqrt(radicand);
    vec2 minusB = vec2(-rDc, -coneB);

    vec4 lambda = vec4(
        (minusB.x - radix.x) / divisor.x,
        (minusB.y + radix.y) / divisor.y,
        (minusB.x + radix.x) / divisor.x,
        (minusB.y - radix.y) / divisor.y);

    bvec4 invalid = bvec4(
        (divisor.x == 0.0) || (radicand.x < 0.0),
        (divisor.y == 0.0) || (radicand.y < 0.0),
        (divisor.x == 0.0) || (radicand.x < 0.0),
        (divisor.y == 0.0) || (radicand.y < 0.0));

    vec4 ix = camPos.xxxx + ray.xxxx * lambda;


    invalid.x = invalid.x || (ix.x < TIP_LEN) || (ix.x > CYL_LEN);
    invalid.y = invalid.y || (ix.y < 0.0) || (ix.y > TIP_LEN);
    invalid.z = invalid.z || !(((ix.z > TIP_LEN) || (ix.x > CYL_LEN)) && (ix.z < CYL_LEN));
    invalid.w = invalid.w || !((ix.w > 0.0) && (ix.w < TIP_LEN));

    if (invalid.x && invalid.y && invalid.z && invalid.w) {
#ifdef CLIP
        discard;
#endif // CLIP
    }

    vec3 intersection, color;
    vec3 normal = vec3(1.0, 0.0, 0.0);
    color = colOut.rgb;

    if (!invalid.y) {
        invalid.xzw = bvec3(true, true, true);
        intersection = camPos.xyz + ray * lambda.y;
        normal = normalize(vec3(-TIP_RAD / TIP_LEN, normalize(intersection.yz)));
//        color = vec3(1.0, 0.0, 0.0);
    }
    if (!invalid.x) {
        invalid.zw = bvec2(true, true);
        intersection = camPos.xyz + ray * lambda.x;
        normal = vec3(0.0, normalize(intersection.yz));
    }
    if (!invalid.z) {
        invalid.w = true;
        lambda.z = (CYL_LEN - camPos.x) / ray.x;
        intersection = camPos.xyz + ray * lambda.z;
    }
    if (!invalid.w) {
        lambda.w = (TIP_LEN - camPos.x) / ray.x;
        intersection = camPos.xyz + ray * lambda.w;
    }

    //color.r = 1.0 - intersection.x / CYL_LEN;
    //color.g = 0.0; // intersection.x / CYL_LEN;
    //color.b = intersection.x / CYL_LEN;

    // phong lighting with directional light
    gl_FragColor = vec4(LocalLighting(ray, normal, light.xyz, color), 1.0);
/*    vec4 lightparams = vec4(0.2, 0.8, 0.4, 10.0);
#define LIGHT_AMBIENT lightparams.x
#define LIGHT_DIFFUSE lightparams.y
#define LIGHT_SPECULAR lightparams.z
#define LIGHT_EXPONENT lightparams.w
    gl_FragColor = vec4(LIGHT_AMBIENT*color, 1.0);*/
    gl_FragColor.rgb = 0.75 * gl_FragColor.rgb + 0.25 * color;


    // calculate depth
#ifdef DEPTH
    tmp = intersection;
    intersection.x = dot(rotMatT0, tmp.xyz);
    intersection.y = dot(rotMatT1, tmp.xyz);
    intersection.z = dot(rotMatT2, tmp.xyz);

    intersection += objPos.xyz;

    vec4 Ding = vec4(intersection, 1.0);
    float depth = dot(modelviewProjTrans[2], Ding);
    float depthW = dot(modelviewProjTrans[3], Ding);
#ifndef CLIP
    if (invalid.x && invalid.y && invalid.z && invalid.w) {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        gl_FragDepth = 0.99999;
    } else {
#endif // CLIP
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
#ifndef CLIP
    }
#endif // CLIP

//    gl_FragColor.rgb *= ;

#endif // DEPTH

#ifdef RETICLE
    coord = gl_FragCoord
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);
    if (min(abs(coord.x - centerFragment.x), abs(coord.y - centerFragment.y)) < 0.002) {
        gl_FragColor.rgb += vec3(0.3, 0.3, 0.5);
    }
#endif // RETICLE*/
}

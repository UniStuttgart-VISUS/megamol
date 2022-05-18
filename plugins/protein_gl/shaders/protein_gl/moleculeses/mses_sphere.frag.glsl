#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"
#include "protein_gl/deferred/gbuffer_output.glsl"
#include "protein_gl/moleculeses/mses_common_defines.glsl"

in vec4 objPos;
in vec4 camPos;
in float move_d;

in float squarRad;
in float rad;
in vec3 move_color;

void main(void) {
    vec4 coord;
    vec3 ray;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform fragment coordinates from view coordinates to object coordinates.
    coord = mvpinverse * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space

    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);

    // calculate the geometry-ray-intersection
    float d1 = -dot(camPos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(camPos.xyz, camPos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = squarRad - d2s;                        // square of difference of projected length and lambda
    
    if (radicand < 0.0) { discard; }

    float lambdaSign = -1.0;

    lambda = d1 + (lambdaSign * sqrt(radicand));                           // lambda
    vec3 sphereintersection = lambda * ray + camPos.xyz;    // intersection point

    // "calc" normal at intersection point
    vec3 normal = sphereintersection / rad;
#ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, move_d);
#endif // SMALL_SPRITE_LIGHTING

    albedo_out = vec4(move_color.rgb,1.0);

    // calculate depth
#ifdef DEPTH
    vec4 Ding = vec4(sphereintersection + objPos.xyz, 1.0);
    float depth = dot(mvptransposed[2], Ding);
    float depthW = dot(mvptransposed[3], Ding);
#ifdef OGL_DEPTH_SES
    float depthval = ((depth / depthW) + 1.0) * 0.5;
#else
    //gl_FragDepth = ( depth + zValues.x) / zValues.y;
    float depthval = (depth + zValues.x)/( zValues.y + zValues.x);
#endif // OGL_DEPTH_SES
#endif // DEPTH

    depth_out = depthval;
    gl_FragDepth = depthval;
    normal_out = normal;
}

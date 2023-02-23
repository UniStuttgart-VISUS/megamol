#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"

#include "protein_gl/simplemolecule/sm_common_input_frag.glsl"
#include "protein_gl/deferred/gbuffer_output.glsl"

in vec3 radz;

in vec3 rotMatT0;
in vec3 rotMatT1; // rotation matrix from the quaternion
in vec3 rotMatT2;

void main(void) {
    vec4 coord;
    vec3 ray, tmp;
    const float maxLambda = 50000.0;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0); 

    // transform fragment coordinates from view coordinates to object coordinates.
    coord = MVPinv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and move

    mat3 rotmat;
    rotmat[0] = rotMatT0;
    rotmat[1] = rotMatT1;
    rotmat[2] = rotMatT2;
    mat3 invrot = inverse(rotmat);

    // calc the viewing ray
    ray = rotmat * coord.xyz;
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

    bool cylPt1Valid = (cylPt1.x <= CYL_HALF_LEN) && (cylPt1.x >= -CYL_HALF_LEN); // trim cylinder
    bool cylPt2Valid = (cylPt2.x <= CYL_HALF_LEN) && (cylPt2.x >= -CYL_HALF_LEN);

    lambdas.x = (cylPt1Valid) ? lambdas.x : lambdas.y;    

#ifdef CLIP
    if ((!cylPt1Valid && !cylPt2Valid)) { discard; }
#else // CLIP
    if ((!cylPt1Valid && !cylPt2Valid)) { radicand = -1.0; }
#endif // CLIP

    if( lambdas.x < 0.0 ) discard;
    vec3 intersection = camPos.xyz + ray * lambdas.x;
    vec3 normal = vec3(0.0, normalize(intersection.yz));
    normal_out = invrot * normal; // inverse the rotation into model space

    // chose color for lighting
    vec3 color = move_color.rgb;

    // set fragment color (kroneml)
    if(cylPt1.x > 0.0 ) {
        color = move_color2.rgb;
    } else {
        color = move_color.rgb;
    }

    if(useClipPlane) {
        // cut with clipping plane
        vec3 planeNormal = normalize(clipPlaneDir);
        // Compute distance of the object from the plane
        float d = -dot(planeNormal, clipPlaneBase - objPos.xyz);
        if (d > 0.0) discard;
    }

    // phong lighting with directional light
    //gl_FragColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    //gl_FragColor = vec4(color.xyz, 1);
    albedo_out = vec4(color.xyz, 1);

    // calculate depth
#ifdef DEPTH
    tmp = intersection;
    intersection.x = dot(rotMatT0, tmp.xyz);
    intersection.y = dot(rotMatT1, tmp.xyz);
    intersection.z = dot(rotMatT2, tmp.xyz);

    intersection += objPos.xyz;

    vec4 Ding = vec4(intersection, 1.0);
    float depth = dot(MVPtransp[2], Ding);
    float depthW = dot(MVPtransp[3], Ding);
#ifndef CLIP
    if (radicand < 0.0) { 
        //gl_FragColor = move_color2;
        albedo_out = move_color2;
        depth = 1.0;
        depthW = 1.0;
    }
#endif // CLIP
    float dval = ((depth / depthW) + 1.0) * 0.5;
    gl_FragDepth = dval;
    depth_out = dval;
#endif // DEPTH
}

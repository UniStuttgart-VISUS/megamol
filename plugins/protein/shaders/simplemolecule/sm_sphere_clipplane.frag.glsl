#version 110

#include "simplemolecule/sm_common_defines.glsl"
#include "lightdirectional.glsl"

#include "simplemolecule/sm_common_input.glsl"

uniform vec3 clipPlaneDir;
uniform vec3 clipPlaneBase;

varying float squarRad;
varying float rad;

void main(void) {
    vec4 coord;
    vec3 ray;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    
    // transform fragment coordinates from view coordinates to object coordinates.
    coord = MVPinv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space

    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);

    // calculate the geometry-ray-intersection
    float d1 = -dot(camPos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(camPos.xyz, camPos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = squarRad - d2s;                        // square of difference of projected length and lambda
#ifdef CLIP
    if (radicand < 0.0) { discard; }
#endif // CLIP
    lambda = d1 - sqrt(radicand);                           // lambda
    vec3 sphereintersection = lambda * ray + camPos.xyz;    // intersection point

    // "calc" normal at intersection point
    vec3 normal = sphereintersection / rad;
#ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, lightPos.w);
#endif // SMALL_SPRITE_LIGHTING

    // chose color for lighting
    vec3 color = gl_Color.rgb;
  
    // cut with clipping plane
    vec3 planeNormal = normalize( clipPlaneDir);
    float d = -dot( planeNormal, clipPlaneBase - objPos.xyz);
    float dist1 = dot( sphereintersection, planeNormal) + d;
    float dist2 = d;
    float t = -( dot( planeNormal, camPos.xyz) + d ) / dot( planeNormal, ray);
    vec3 planeintersect = camPos.xyz + t * ray;
    if( dist1 > 0.0 )
    {
        if( dist2 < rad )
        {
            if( length( planeintersect) < rad )
            {
                sphereintersection = planeintersect;
                normal = planeNormal;
                color *= 0.6;
            }
            else
            {
                discard;
            }
        }
        else
        {
            discard;
        }
    }

    // phong lighting with directional light
    gl_FragColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    
    // calculate depth
#ifdef DEPTH
    vec4 Ding = vec4(sphereintersection + objPos.xyz, 1.0);
    float depth = dot(MVPtransp[2], Ding);
    float depthW = dot(MVPtransp[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
#ifndef CLIP
    gl_FragDepth = (radicand < 0.0) ? 1.0 : ((depth / depthW) + 1.0) * 0.5;
    gl_FragColor.rgb = (radicand < 0.0) ? gl_Color.rgb : gl_FragColor.rgb;
#endif // CLIP
#endif // DEPTH
}
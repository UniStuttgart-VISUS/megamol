#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"
#include "protein_gl/simplemolecule/sm_common_input_frag.glsl"
#include "protein_gl/deferred/gbuffer_output.glsl"

in float squarRad;
in float rad;

void main(void) {

    vec4 coord;
    vec3 ray;
    float lambda;
    vec3 color = vec3(0);
    vec3 sphereintersection = vec3(0.0);
    vec3 normal = vec3(0);

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
    lambda = d1 - sqrt(radicand);                           // lambda

    float radicand2 = 0.0;
    if( radicand < 0.0 ) {
        discard;
    }
    else {
        // chose color for lighting
        color = move_color.rgb;
        if( lambda < 0.0 ) discard;
        sphereintersection = lambda * ray + camPos.xyz;    // intersection point
        // "calc" normal at intersection point
        normal = sphereintersection / rad;
    }
    
#endif // CLIP

#ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, lightPos.w);
#endif // SMALL_SPRITE_LIGHTING

    if(useClipPlane) {
        vec3 planeNormal = normalize( clipPlaneDir);
        float d = -dot(planeNormal, clipPlaneBase - objPos.xyz);
        float dist1 = dot(sphereintersection, planeNormal) + d;
        float dist2 = d;
        float t = -(dot( planeNormal, camPos.xyz) + d) / dot(planeNormal, ray);
        vec3 planeintersect = camPos.xyz + t * ray;
        if(dist1 > 0.0) { 
            if(dist2 < rad) {
                if( length(planeintersect) < rad ) {
                    sphereintersection = planeintersect;
                    normal = planeNormal;
                    color *= 0.6;
                } else {
                    discard;
                }
            } else {
                discard;
            }
        }
    }

    // phong lighting with directional light
    //gl_FragColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    // gl_FragColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color), gl_Color.w);
    //gl_FragColor = vec4(color, 1);
    albedo_out = vec4(color, 1);
    gl_FragDepth = gl_FragCoord.z;
    depth_out = gl_FragCoord.z;
    normal_out = normal;

    // calculate depth
#ifdef DEPTH
    vec4 Ding = vec4(sphereintersection + objPos.xyz, 1.0);
    float depth = dot(MVPtransp[2], Ding);
    float depthW = dot(MVPtransp[3], Ding);
    float depthval = ((depth / depthW) + 1.0) * 0.5; 
    gl_FragDepth = depthval;
    depth_out = depthval;
    
#ifndef CLIP
    gl_FragDepth = (radicand < 0.0) ? 1.0 : depthval;
    depth_out = (radicand < 0.0) ? 1.0 : depthval;
    //gl_FragColor = vec4(color, 1);
    albedo_out = vec4(color, 1);
#endif // CLIP

#endif // DEPTH

    //gl_FragColor.rgb = normal;
    //gl_FragColor.rgb = lightPos.xyz;
}

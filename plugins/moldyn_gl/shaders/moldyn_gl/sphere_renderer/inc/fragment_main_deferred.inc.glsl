#include "moldyn_gl/sphere_renderer/inc/fragment_extensions.inc.glsl"
#include "commondefines.glsl" //remove from sphere renderer eventually?
#include "moldyn_gl/sphere_renderer/inc/fragment_attributes_in.inc.glsl"
#include "moldyn_gl/sphere_renderer/inc/fragment_attributes_out_deferred.inc.glsl"
#include "moldyn_gl/sphere_renderer/inc/fragment_uniforms.inc.glsl"
#include "moldyn_gl/sphere_renderer/inc/fragment_functions.inc.glsl"
#include "lightdirectional.glsl"

void main(void) {

    vec3 ray = computeRay(gl_FragCoord,viewAttr,MVPinv,camPos,objPos);
    Intersection sphereIntersection = computeRaySphereIntersection(ray,camPos,objPos,vertColor,squarRad,rad,clipDat,clipCol);

    // "calc" normal at intersection point
#ifdef SMALL_SPRITE_LIGHTING
    sphereIntersection.normal = mix(-ray, sphereIntersection.normal, outlightDir.w);
#endif // SMALL_SPRITE_LIGHTING

#ifdef AXISHINTS
    sphereIntersection.color = axisHintsColor(normal);
#endif // AXISHINTS

    // Output unlit surface color and normal to render targets
    outColor = sphereIntersection.color;
    outNormal = sphereIntersection.normal;

// Calculate depth
#ifdef DEPTH

    float depth = computeDepthValue(sphereIntersection.position + objPos.xyz,MVPtransp);
    gl_FragDepth = depth;

#ifndef CLIP
    gl_FragDepth = (delta < 0.0) ? 1.0 : depth;
    outColor.rgb = (delta < 0.0) ? vertColor.rgb : outColor.rgb;
#endif // CLIP

#ifdef DISCARD_COLOR_MARKER
    gl_FragDepth = computeDepthValue(objPos.xyz, MVPtransp);
#endif // DISCARD_COLOR_MARKER

#endif // DEPTH

#ifdef RETICLE
    outColor.rgb = reticleColor(outColor.rgb, gl_FragColor, sphere_frag_center);
#endif // RETICLE

}

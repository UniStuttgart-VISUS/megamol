#version 330

#include "moldyn_gl/sphere_renderer/inc/fragment_extensions.inc.glsl"
#include "commondefines.glsl" //remove from sphere renderer eventually?
#include "moldyn_gl/sphere_renderer/inc/fragment_attributes_in.inc.glsl"
#include "moldyn_gl/sphere_renderer/inc/fragment_attributes_out.inc.glsl"
#include "moldyn_gl/sphere_renderer/inc/fragment_uniforms.inc.glsl"
#include "moldyn_gl/sphere_renderer/inc/fragment_functions.inc.glsl"
#include "lightdirectional.glsl"

void main(void) {

    vec4 color = vertColor;
    float distance = length(gl_FragCoord.xy - sphere_frag_center);

#ifdef CLIP

    if (distance > (sphere_frag_radius + outlineWidth)) {
#ifdef DISCARD_COLOR_MARKER
        color = vec4(1.0, 0.0, 0.0, 1.0);
#else // DISCARD_COLOR_MARKER
        discard;
#endif // DISCARD_COLOR_MARKER
    }

#endif // CLIP

    if (length(gl_FragCoord.xy - sphere_frag_center) < sphere_frag_radius) {
        discard;
    }
    outColor = color;

// Calculate depth
#ifdef DEPTH

    gl_FragDepth = computeDepthValue(objPos.xyz, MVPtransp);

#endif // DEPTH

#ifdef RETICLE
    outColor.rgb = reticleColor(outColor.rgb, gl_FragColor, sphere_frag_center);
#endif // RETICLE

}

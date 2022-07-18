#version 430

#include "moldyn_gl/sphere_renderer/inc/mdao_deferred_fragment_main.inc.glsl"
#ifdef ENABLE_LIGHTING
#include "lightdirectional.glsl"
#else
#include "moldyn_gl/sphere_renderer/inc/mdao_deferred_fragment_lighting_stub.inc.glsl"
#endif
#include "moldyn_gl/sphere_renderer/inc/mdao_deferred_directions.inc.glsl"
#include "moldyn_gl/sphere_renderer/inc/mdao_deferred_fragment_ambocc.inc.glsl"

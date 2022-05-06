#version 430

#include "sphere_renderer/inc/mdao_deferred_fragment_main.inc.glsl"
#ifdef ENABLE_LIGHTING
#include "lightdirectional.glsl"
#else
#include "sphere_renderer/inc/mdao_deferred_fragment_lighting_stub.inc.glsl"
#endif
#include "sphere_renderer/inc/mdao_deferred_directions.inc.glsl"
#include "sphere_renderer/inc/mdao_deferred_fragment_ambocc.inc.glsl"

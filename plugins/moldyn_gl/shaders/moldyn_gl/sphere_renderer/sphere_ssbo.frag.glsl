#version 330

#ifdef DEFERRED_SHADING
#include "moldyn_gl/sphere_renderer/inc/fragment_main_deferred.inc.glsl"
#else
#include "moldyn_gl/sphere_renderer/inc/fragment_main.inc.glsl"
#endif

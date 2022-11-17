#version 430

#include "pwdemos_gl/quartz/snippets/ray_frag_1.glsl"
#include "pwdemos_gl/quartz/snippets/common_directLight.glsl"
#include "pwdemos_gl/quartz/snippets/ray_frag_2.glsl"
#include "pwdemos_gl/quartz/snippets/ray_frag_3.glsl"
#ifdef REPLACE_FRAG_SNIPPET
REPLACE_FRAG_SNIPPET
#else
#include "pwdemos_gl/quartz/snippets/ray_frag_mastercaster.glsl"
#endif
#include "pwdemos_gl/quartz/snippets/ray_frag_4.glsl"

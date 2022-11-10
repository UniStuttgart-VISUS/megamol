#version 430

#include "pwdemos_gl/quartz/snippets/ray_fragclipped_1.glsl"
#include "pwdemos_gl/quartz/snippets/common_directLight.glsl"
#include "pwdemos_gl/quartz/snippets/ray_fragclipped_2.glsl"
#include "pwdemos_gl/quartz/snippets/ray_fragclipped_3.glsl"
#ifdef REPLACE_FRAG_SNIPPET
REPLACE_FRAG_SNIPPET
#else
#include" pwdemos_gl/quartz/snippets/ray_fragclipped_mastercaster.glsl"
#endif
#include "pwdemos_gl/quartz/snippets/ray_fragclipped_4.glsl"
#include "pwdemos_gl/quartz/snippets/ray_fragclipped_5.glsl"
#include "pwdemos_gl/quartz/snippets/ray_fragclipped_6.glsl"

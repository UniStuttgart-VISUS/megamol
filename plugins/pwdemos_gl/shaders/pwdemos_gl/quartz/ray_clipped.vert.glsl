#version 430

#ifdef REPLACE_VERT_SNIPPET
#define OUTERRAD REPLACE_VERT_SNIPPET
#else
#include "pwdemos_gl/quartz/snippets/ray_vertclipped_replaceMeParams.glsl"
#endif
#include "pwdemos_gl/quartz/snippets/ray_vertclipped.glsl"

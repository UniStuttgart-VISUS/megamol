#version 130

#include "common_defines_btf.glsl"
#include "protein_cuda/tubetessellation/vertex_attributes.glsl"
#include "protein_cuda/tubetessellation/vertex_MainParams.glsl"
// here comes the injected snippet
#include "protein_cuda/tubetessellation/vertex_MainRest.glsl"
#include "protein_cuda/tubetessellation/vertex_posTrans.glsl"

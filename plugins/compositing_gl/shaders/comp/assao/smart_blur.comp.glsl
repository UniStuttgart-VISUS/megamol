/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

#include "uniforms/assao_constants.inc.glsl"
#include "uniforms/blur_uniforms.inc.glsl"
#include "calc/add_sample.inc.glsl"
#include "calc/unpack_edges.inc.glsl"
#include "calc/sample_blurred.inc.glsl"
#include "calc/main_smart_blur.inc.glsl"

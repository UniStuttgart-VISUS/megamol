/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

#include "uniforms/assao_constants.inc.glsl"
#include "uniforms/prepare_depths_and_normals_uniforms.inc.glsl"
#include "calc/screenspace_to_viewspace_depth.inc.glsl"
#include "calc/calculate_edges.inc.glsl"
#include "calc/ndc_to_viewspace.inc.glsl"
#include "calc/calculate_normal.inc.glsl"
#include "calc/main_prepare_depths_and_normals.inc.glsl"

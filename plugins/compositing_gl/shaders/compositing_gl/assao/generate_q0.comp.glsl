/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

#include "uniforms/assao_constants.inc.glsl"
#include "uniforms/generate_qx_uniforms.inc.glsl"
#include "calc/pack_edges.inc.glsl"
#include "calc/ndc_to_viewspace.inc.glsl"
#include "calc/decode_normal.inc.glsl"
#include "calc/load_normal.inc.glsl"
#include "calc/load_normal_offset.inc.glsl"
#include "calc/calculate_radius_parameters.inc.glsl"
#include "calc/calculate_edges.inc.glsl"
#include "calc/calculate_pixel_obscurance.inc.glsl"
#include "calc/ssao_tap_inner.inc.glsl"
#include "calc/ssao_tap.inc.glsl"
#include "calc/generate_ssao_shadows_internal.inc.glsl"
#include "calc/main_generate_q0.inc.glsl"

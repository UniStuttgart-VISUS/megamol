/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

// ** WARNING ** if changing anything here, please remember to update the corresponding C++ code!
struct SMAAConstants
{
    float SMAA_THRESHOLD;
    float SMAA_DEPTH_THRESHOLD;
    int   SMAA_MAX_SEARCH_STEPS;
    int   SMAA_MAX_SEARCH_STEPS_DIAG;

    int   SMAA_CORNER_ROUNDING;
    float SMAA_CORNER_ROUNDING_NORM;
    float SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR;
    int   SMAA_PREDICATION;

    float SMAA_PREDICATION_THRESHOLD;
    float SMAA_PREDICATION_SCALE;
    float SMAA_PREDICATION_STRENGTH;
    int   SMAA_REPROJECTION;

    float SMAA_REPROJECTION_WEIGHT_SCALE;
    int   SMAA_DISABLE_DIAG_DETECTION;
    int   SMAA_DISABLE_CORNER_DETECTION;
    int   SMAA_DECODE_VELOCITY;

    vec4  SMAA_RT_METRICS;
};

uniform vec4 rt_snd;

//-----------------------------------------------------------------------------
// Non-Configurable Defines
#define SMAA_AREATEX_MAX_DISTANCE 16
#define SMAA_AREATEX_MAX_DISTANCE_DIAG 20
#define SMAA_AREATEX_PIXEL_SIZE (1.0 / vec2(160.0, 560.0))
#define SMAA_AREATEX_SUBTEX_SIZE (1.0 / 7.0)
#define SMAA_SEARCHTEX_SIZE vec2(66.0, 33.0)
#define SMAA_SEARCHTEX_PACKED_SIZE vec2(64.0, 16.0)

layout( std430, binding = 0 ) readonly buffer SMAAConstansBuffer {
    SMAAConstants g_SMAAConsts;
};

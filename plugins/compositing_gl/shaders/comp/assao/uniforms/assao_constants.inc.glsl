///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

// progressive poisson-like pattern; x, y are in [-1, 1] range, .z is length( vec2(x,y) ), .w is log2( z )
#define INTELSSAO_MAIN_DISK_SAMPLE_COUNT (32)
const vec4 g_samplePatternMain[INTELSSAO_MAIN_DISK_SAMPLE_COUNT] =
{
    vec4( 0.78488064,  0.56661671,  1.500000, -0.126083),    vec4( 0.26022232, -0.29575172,  1.500000, -1.064030),    vec4( 0.10459357,  0.08372527,  1.110000, -2.730563),    vec4(-0.68286800,  0.04963045,  1.090000, -0.498827),
    vec4(-0.13570161, -0.64190155,  1.250000, -0.532765),    vec4(-0.26193795, -0.08205118,  0.670000, -1.783245),    vec4(-0.61177456,  0.66664219,  0.710000, -0.044234),    vec4( 0.43675563,  0.25119025,  0.610000, -1.167283),
    vec4( 0.07884444,  0.86618668,  0.640000, -0.459002),    vec4(-0.12790935, -0.29869005,  0.600000, -1.729424),    vec4(-0.04031125,  0.02413622,  0.600000, -4.792042),    vec4( 0.16201244, -0.52851415,  0.790000, -1.067055),
    vec4(-0.70991218,  0.47301072,  0.640000, -0.335236),    vec4( 0.03277707, -0.22349690,  0.600000, -1.982384),    vec4( 0.68921727,  0.36800742,  0.630000, -0.266718),    vec4( 0.29251814,  0.37775412,  0.610000, -1.422520),
    vec4(-0.12224089,  0.96582592,  0.600000, -0.426142),    vec4( 0.11071457, -0.16131058,  0.600000, -2.165947),    vec4( 0.46562141, -0.59747696,  0.600000, -0.189760),    vec4(-0.51548797,  0.11804193,  0.600000, -1.246800),
    vec4( 0.89141309, -0.42090443,  0.600000,  0.028192),    vec4(-0.32402530, -0.01591529,  0.600000, -1.543018),    vec4( 0.60771245,  0.41635221,  0.600000, -0.605411),    vec4( 0.02379565, -0.08239821,  0.600000, -3.809046),
    vec4( 0.48951152, -0.23657045,  0.600000, -1.189011),    vec4(-0.17611565, -0.81696892,  0.600000, -0.513724),    vec4(-0.33930185, -0.20732205,  0.600000, -1.698047),    vec4(-0.91974425,  0.05403209,  0.600000,  0.062246),
    vec4(-0.15064627, -0.14949332,  0.600000, -1.896062),    vec4( 0.53180975, -0.35210401,  0.600000, -0.758838),    vec4( 0.41487166,  0.81442589,  0.600000, -0.505648),    vec4(-0.24106961, -0.32721516,  0.600000, -1.665244)
};

// these values can be changed (up to SSAO_MAX_TAPS) with no changes required elsewhere; values for 4th and 5th preset are ignored but array needed to avoid compilation errors
// the actual number of texture samples is two times this value (each "tap" has two symmetrical depth texture samples)
const uint g_numTaps[5]   = { 3, 5, 12, 0, 0 };

// an example of higher quality low/medium/high settings
// static const uint g_numTaps[5]   = { 4, 9, 16, 0, 0 };

// ** WARNING ** if changing anything here, please remember to update the corresponding C++ code!
struct ASSAOConstants
{
    vec2  ViewportPixelSize;                      // .zw == 1.0 / ViewportSize.xy
    vec2  HalfViewportPixelSize;                  // .zw == 1.0 / ViewportHalfSize.xy

    vec2  DepthUnpackConsts;
    vec2  CameraTanHalfFOV;

    vec2  NDCToViewMul;
    vec2  NDCToViewAdd;

    ivec2 PerPassFullResCoordOffset;
    vec2  PerPassFullResUVOffset;

    vec2  Viewport2xPixelSize;
    vec2  Viewport2xPixelSize_x_025;              // Viewport2xPixelSize * 0.25 (for fusing add+mul into mad)

    float EffectRadius;                           // world (viewspace) maximum size of the shadow
    float EffectShadowStrength;                   // global strength of the effect (0 - 5)
    float EffectShadowPow;
    float EffectShadowClamp;

    float EffectFadeOutMul;                       // effect fade out from distance (ex. 25)
    float EffectFadeOutAdd;                       // effect fade out to distance   (ex. 100)
    float EffectHorizonAngleThreshold;            // limit errors on slopes and caused by insufficient geometry tessellation (0.05 to 0.5)
    float EffectSamplingRadiusNearLimitRec;       // if viewspace pixel closer than this, don't enlarge shadow sampling radius anymore (makes no sense to grow beyond some distance, not enough samples to cover everything, so just limit the shadow growth; could be SSAOSettingsFadeOutFrom * 0.1 or less)

    float DepthPrecisionOffsetMod;
    float NegRecEffectRadius;                     // -1.0 / EffectRadius
    float LoadCounterAvgDiv;                      // 1.0 / ( halfDepthMip[SSAO_DEPTH_MIP_LEVELS-1].sizeX * halfDepthMip[SSAO_DEPTH_MIP_LEVELS-1].sizeY )
    float AdaptiveSampleCountLimit;

    float InvSharpness;
    int   PassIndex;
    vec2  QuarterResPixelSize;                    // used for importance map only

    vec4  PatternRotScaleMatrices[5];

    float NormalsUnpackMul;
    float NormalsUnpackAdd;
    float DetailAOStrength;
    int TransformNormalsToViewSpace;

#if SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
    mat4  NormalsWorldToViewspaceMatrix;
#endif

    mat4 ViewMX;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Optional parts that can be enabled for a required quality preset level and above (0 == Low, 1 == Medium, 2 == High, 3 == Highest/Adaptive, 4 == reference/unused )
// Each has its own cost. To disable just set to 5 or above.
//
// (experimental) tilts the disk (although only half of the samples!) towards surface normal; this helps with effect uniformity between objects but reduces effect distance and has other side-effects
#define SSAO_TILT_SAMPLES_ENABLE_AT_QUALITY_PRESET                      (99)        // to disable simply set to 99 or similar
#define SSAO_TILT_SAMPLES_AMOUNT                                        (0.4)
//
#define SSAO_HALOING_REDUCTION_ENABLE_AT_QUALITY_PRESET                 (1)         // to disable simply set to 99 or similar
#define SSAO_HALOING_REDUCTION_AMOUNT                                   (0.6)       // values from 0.0 - 1.0, 1.0 means max weighting (will cause artifacts, 0.8 is more reasonable)
//
#define SSAO_NORMAL_BASED_EDGES_ENABLE_AT_QUALITY_PRESET                (2)         // to disable simply set to 99 or similar
#define SSAO_NORMAL_BASED_EDGES_DOT_THRESHOLD                           (0.5)       // use 0-0.1 for super-sharp normal-based edges
//
#define SSAO_DETAIL_AO_ENABLE_AT_QUALITY_PRESET                         (1)         // whether to use DetailAOStrength; to disable simply set to 99 or similar
//
#define SSAO_DEPTH_MIPS_ENABLE_AT_QUALITY_PRESET                        (2)         // !!warning!! the MIP generation on the C++ side will be enabled on quality preset 2 regardless of this value, so if changing here, change the C++ side too
#define SSAO_DEPTH_MIPS_GLOBAL_OFFSET                                   (-4.3)      // best noise/quality/performance tradeoff, found empirically
//
// !!warning!! the edge handling is hard-coded to 'disabled' on quality level 0, and enabled above, on the C++ side; while toggling it here will work for
// testing purposes, it will not yield performance gains (or correct results)
#define SSAO_DEPTH_BASED_EDGES_ENABLE_AT_QUALITY_PRESET                 (1)
//
#define SSAO_REDUCE_RADIUS_NEAR_SCREEN_BORDER_ENABLE_AT_QUALITY_PRESET  (99)        // 99 means disabled; only helpful if artifacts at the edges caused by lack of out of screen depth data are not acceptable with the depth sampler in either clamp or mirror modes
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

layout( std430, binding = 0 ) readonly buffer SSAOConstantsBuffer {
    ASSAOConstants g_ASSAOConsts;
};

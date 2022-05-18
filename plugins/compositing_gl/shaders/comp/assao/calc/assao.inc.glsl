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
// File changes (yyyy-mm-dd)
// 2016-09-07: filip.strugar@intel.com: first commit
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// progressive poisson-like pattern; x, y are in [-1, 1] range, .z is length( vec2(x,y) ), .w is log2( z )
#define INTELSSAO_MAIN_DISK_SAMPLE_COUNT (32)
static const vec4 g_samplePatternMain[INTELSSAO_MAIN_DISK_SAMPLE_COUNT] =
{
     0.78488064,  0.56661671,  1.500000, -0.126083,     0.26022232, -0.29575172,  1.500000, -1.064030,     0.10459357,  0.08372527,  1.110000, -2.730563,    -0.68286800,  0.04963045,  1.090000, -0.498827,
    -0.13570161, -0.64190155,  1.250000, -0.532765,    -0.26193795, -0.08205118,  0.670000, -1.783245,    -0.61177456,  0.66664219,  0.710000, -0.044234,     0.43675563,  0.25119025,  0.610000, -1.167283,
     0.07884444,  0.86618668,  0.640000, -0.459002,    -0.12790935, -0.29869005,  0.600000, -1.729424,    -0.04031125,  0.02413622,  0.600000, -4.792042,     0.16201244, -0.52851415,  0.790000, -1.067055,
    -0.70991218,  0.47301072,  0.640000, -0.335236,     0.03277707, -0.22349690,  0.600000, -1.982384,     0.68921727,  0.36800742,  0.630000, -0.266718,     0.29251814,  0.37775412,  0.610000, -1.422520,
    -0.12224089,  0.96582592,  0.600000, -0.426142,     0.11071457, -0.16131058,  0.600000, -2.165947,     0.46562141, -0.59747696,  0.600000, -0.189760,    -0.51548797,  0.11804193,  0.600000, -1.246800,
     0.89141309, -0.42090443,  0.600000,  0.028192,    -0.32402530, -0.01591529,  0.600000, -1.543018,     0.60771245,  0.41635221,  0.600000, -0.605411,     0.02379565, -0.08239821,  0.600000, -3.809046,
     0.48951152, -0.23657045,  0.600000, -1.189011,    -0.17611565, -0.81696892,  0.600000, -0.513724,    -0.33930185, -0.20732205,  0.600000, -1.698047,    -0.91974425,  0.05403209,  0.600000,  0.062246,
    -0.15064627, -0.14949332,  0.600000, -1.896062,     0.53180975, -0.35210401,  0.600000, -0.758838,     0.41487166,  0.81442589,  0.600000, -0.505648,    -0.24106961, -0.32721516,  0.600000, -1.665244
};

// these values can be changed (up to SSAO_MAX_TAPS) with no changes required elsewhere; values for 4th and 5th preset are ignored but array needed to avoid compilation errors
// the actual number of texture samples is two times this value (each "tap" has two symmetrical depth texture samples)
static const uint g_numTaps[5]   = { 3, 5, 12, 0, 0 };

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
    float Dummy0;

#if SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
    mat4  NormalsWorldToViewspaceMatrix;
#endif
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

cbuffer SSAOConstantsBuffer                     : register( b0 )    // corresponds to SSAO_CONSTANTS_BUFFERSLOT
{
    ASSAOConstants                    g_ASSAOConsts;
}

SamplerState        g_PointClampSampler         : register( s0 );   // corresponds to SSAO_SAMPLERS_SLOT0
SamplerState        g_LinearClampSampler        : register( s1 );   // corresponds to SSAO_SAMPLERS_SLOT1
SamplerState        g_PointMirrorSampler        : register( s2 );   // corresponds to SSAO_SAMPLERS_SLOT2
SamplerState        g_ViewspaceDepthTapSampler  : register( s3 );   // corresponds to SSAO_SAMPLERS_SLOT3

Texture2D<float>    g_DepthSource               : register( t0 );   // corresponds to SSAO_TEXTURE_SLOT0
Texture2D           g_NormalmapSource           : register( t1 );   // corresponds to SSAO_TEXTURE_SLOT1

Texture2D<float>    g_ViewspaceDepthSource      : register( t0 );   // corresponds to SSAO_TEXTURE_SLOT0
Texture2D<float>    g_ViewspaceDepthSource1     : register( t1 );   // corresponds to SSAO_TEXTURE_SLOT1
Texture2D<float>    g_ViewspaceDepthSource2     : register( t2 );   // corresponds to SSAO_TEXTURE_SLOT2
Texture2D<float>    g_ViewspaceDepthSource3     : register( t3 );   // corresponds to SSAO_TEXTURE_SLOT3

Texture2D<float>    g_ImportanceMap             : register( t3 );   // corresponds to SSAO_TEXTURE_SLOT3

Texture1D<uint>     g_LoadCounter               : register( t2 );   // corresponds to SSAO_TEXTURE_SLOT2

Texture2D           g_BlurInput                 : register( t2 );   // corresponds to SSAO_TEXTURE_SLOT2

Texture2DArray      g_FinalSSAO                 : register( t4 );   // corresponds to SSAO_TEXTURE_SLOT4

RWTexture2D<unorm vec4> g_NormalsOutputUAV    : register( u4 );   // corresponds to SSAO_NORMALMAP_OUT_UAV_SLOT
RWTexture1D<uint>   g_LoadCounterOutputUAV      : register( u4 );   // corresponds to SSAO_LOAD_COUNTER_UAV_SLOT


// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
float PackEdges( vec4 edgesLRTB )
{
//    ivec4 edgesLRTBi = ivec4( saturate( edgesLRTB ) * 3.0 + 0.5 );
//    return ( (edgesLRTBi.x << 6) + (edgesLRTBi.y << 4) + (edgesLRTBi.z << 2) + (edgesLRTBi.w << 0) ) / 255.0;

    // optimized, should be same as above
    edgesLRTB = round( clamp( edgesLRTB, 0.0, 1.0 ) ) * 3.05 );
    return dot( edgesLRTB, vec4( 64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0 ) ) ;
}

vec4 UnpackEdges( float _packedVal )
{
    uint packedVal = (uint)(_packedVal * 255.5);
    vec4 edgesLRTB;
    edgesLRTB.x = float((packedVal >> 6) & 0x03) / 3.0;          // there's really no need for mask (as it's an 8 bit input) but I'll leave it in so it doesn't cause any trouble in the future
    edgesLRTB.y = float((packedVal >> 4) & 0x03) / 3.0;
    edgesLRTB.z = float((packedVal >> 2) & 0x03) / 3.0;
    edgesLRTB.w = float((packedVal >> 0) & 0x03) / 3.0;

    return clamp( edgesLRTB + g_ASSAOConsts.InvSharpness, 0.0, 1.0 ); // saturate(x) == clamp(x, 0, 1)
}

float ScreenSpaceToViewSpaceDepth( float screenDepth )
{
    float depthLinearizeMul = g_ASSAOConsts.DepthUnpackConsts.x;
    float depthLinearizeAdd = g_ASSAOConsts.DepthUnpackConsts.y;

    // Optimised version of "-cameraClipNear / (cameraClipFar - projDepth * (cameraClipFar - cameraClipNear)) * cameraClipFar"

    // Set your depthLinearizeMul and depthLinearizeAdd to:
    // depthLinearizeMul = ( cameraClipFar * cameraClipNear) / ( cameraClipFar - cameraClipNear );
    // depthLinearizeAdd = cameraClipFar / ( cameraClipFar - cameraClipNear );

    return depthLinearizeMul / ( depthLinearizeAdd - screenDepth );
}

// from [0, width], [0, height] to [-1, 1], [-1, 1]
vec2 ScreenSpaceToClipSpacePositionXY( vec2 screenPos )
{
    return screenPos * g_ASSAOConsts.Viewport2xPixelSize.xy - vec2( 1.0f, 1.0f );
}

vec3 ScreenSpaceToViewSpacePosition( vec2 screenPos, float viewspaceDepth )
{
    return vec3( g_ASSAOConsts.CameraTanHalfFOV.xy * viewspaceDepth * ScreenSpaceToClipSpacePositionXY( screenPos ), viewspaceDepth );
}

vec3 ClipSpaceToViewSpacePosition( vec2 clipPos, float viewspaceDepth )
{
    return vec3( g_ASSAOConsts.CameraTanHalfFOV.xy * viewspaceDepth * clipPos, viewspaceDepth );
}

vec3 NDCToViewspace( vec2 pos, float viewspaceDepth )
{
    vec3 ret;

    ret.xy = (g_ASSAOConsts.NDCToViewMul * pos.xy + g_ASSAOConsts.NDCToViewAdd) * viewspaceDepth;

    ret.z = viewspaceDepth;

    return ret;
}

// calculate effect radius and fit our screen sampling pattern inside it
void CalculateRadiusParameters( const float pixCenterLength, const vec2 pixelDirRBViewspaceSizeAtCenterZ, out float pixLookupRadiusMod, out float effectRadius, out float falloffCalcMulSq )
{
    effectRadius = g_ASSAOConsts.EffectRadius;

    // leaving this out for performance reasons: use something similar if radius needs to scale based on distance
    //effectRadius *= pow( pixCenterLength, g_ASSAOConsts.RadiusDistanceScalingFunctionPow);

    // when too close, on-screen sampling disk will grow beyond screen size; limit this to avoid closeup temporal artifacts
    const float tooCloseLimitMod = clamp( pixCenterLength * g_ASSAOConsts.EffectSamplingRadiusNearLimitRec, 0.0, 1.0 ) * 0.8 + 0.2;

    effectRadius *= tooCloseLimitMod;

    // 0.85 is to reduce the radius to allow for more samples on a slope to still stay within influence
    pixLookupRadiusMod = (0.85 * effectRadius) / pixelDirRBViewspaceSizeAtCenterZ.x;

    // used to calculate falloff (both for AO samples and per-sample weights)
    falloffCalcMulSq= -1.0f / (effectRadius*effectRadius);
}

vec4 CalculateEdges( const float centerZ, const float leftZ, const float rightZ, const float topZ, const float bottomZ )
{
    // slope-sensitive depth-based edge detection
    vec4 edgesLRTB = vec4( leftZ, rightZ, topZ, bottomZ ) - centerZ;
    vec4 edgesLRTBSlopeAdjusted = edgesLRTB + edgesLRTB.yxwz;
    edgesLRTB = min( abs( edgesLRTB ), abs( edgesLRTBSlopeAdjusted ) );
    return clamp( ( 1.3 - edgesLRTB / (centerZ * 0.040) ), 0.0, 1.0 );

    // cheaper version but has artifacts
    // edgesLRTB = abs( vec4( leftZ, rightZ, topZ, bottomZ ) - centerZ; );
    // return saturate( ( 1.3 - edgesLRTB / (pixZ * 0.06 + 0.1) ) );
}

// pass-through vertex shader
void VSMain( inout vec4 Pos : SV_POSITION, inout vec4 Uv : TEXCOORD0 ) { }

void PSPrepareDepths( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
{
#if 0   // gather can be a bit faster but doesn't work with input depth buffers that don't match the working viewport
    vec2 gatherUV = inPos.xy * g_ASSAOConsts.Viewport2xPixelSize;
    vec4 depths = g_DepthSource.GatherRed( g_PointClampSampler, gatherUV );
    float a = depths.w;  // g_DepthSource.Load( ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 0, 0 ) ).x;
    float b = depths.z;  // g_DepthSource.Load( ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 1, 0 ) ).x;
    float c = depths.x;  // g_DepthSource.Load( ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 0, 1 ) ).x;
    float d = depths.y;  // g_DepthSource.Load( ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 1, 1 ) ).x;
#else
    ivec3 baseCoord = ivec3( ivec2(inPos.xy) * 2, 0 );
    float a = g_DepthSource.Load( baseCoord, ivec2( 0, 0 ) ).x;
    float b = g_DepthSource.Load( baseCoord, ivec2( 1, 0 ) ).x;
    float c = g_DepthSource.Load( baseCoord, ivec2( 0, 1 ) ).x;
    float d = g_DepthSource.Load( baseCoord, ivec2( 1, 1 ) ).x;
#endif

    out0 = ScreenSpaceToViewSpaceDepth( a );
    out1 = ScreenSpaceToViewSpaceDepth( b );
    out2 = ScreenSpaceToViewSpaceDepth( c );
    out3 = ScreenSpaceToViewSpaceDepth( d );
}

void PSPrepareDepthsHalf( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1 )
{
    ivec3 baseCoord = ivec3( ivec2(inPos.xy) * 2, 0 );
    float a = g_DepthSource.Load( baseCoord, ivec2( 0, 0 ) ).x;
    float d = g_DepthSource.Load( baseCoord, ivec2( 1, 1 ) ).x;

    out0 = ScreenSpaceToViewSpaceDepth( a );
    out1 = ScreenSpaceToViewSpaceDepth( d );
}

vec3 CalculateNormal( const vec4 edgesLRTB, vec3 pixCenterPos, vec3 pixLPos, vec3 pixRPos, vec3 pixTPos, vec3 pixBPos )
{
    // Get this pixel's viewspace normal
    vec4 acceptedNormals  = vec4( edgesLRTB.x*edgesLRTB.z, edgesLRTB.z*edgesLRTB.y, edgesLRTB.y*edgesLRTB.w, edgesLRTB.w*edgesLRTB.x );

    pixLPos = normalize(pixLPos - pixCenterPos);
    pixRPos = normalize(pixRPos - pixCenterPos);
    pixTPos = normalize(pixTPos - pixCenterPos);
    pixBPos = normalize(pixBPos - pixCenterPos);

    vec3 pixelNormal = vec3( 0, 0, -0.0005 );
    pixelNormal += ( acceptedNormals.x ) * cross( pixLPos, pixTPos );
    pixelNormal += ( acceptedNormals.y ) * cross( pixTPos, pixRPos );
    pixelNormal += ( acceptedNormals.z ) * cross( pixRPos, pixBPos );
    pixelNormal += ( acceptedNormals.w ) * cross( pixBPos, pixLPos );
    pixelNormal = normalize( pixelNormal );

    return pixelNormal;
}

void PSPrepareDepthsAndNormals( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
{
    ivec2 baseCoords = (( ivec2 )inPos.xy) * 2;
    vec2 upperLeftUV = (inPos.xy - 0.25) * g_ASSAOConsts.Viewport2xPixelSize;

#if 0   // gather can be a bit faster but doesn't work with input depth buffers that don't match the working viewport
    vec2 gatherUV = inPos.xy * g_ASSAOConsts.Viewport2xPixelSize;
    vec4 depths = g_DepthSource.GatherRed( g_PointClampSampler, gatherUV );
    out0 = ScreenSpaceToViewSpaceDepth( depths.w );
    out1 = ScreenSpaceToViewSpaceDepth( depths.z );
    out2 = ScreenSpaceToViewSpaceDepth( depths.x );
    out3 = ScreenSpaceToViewSpaceDepth( depths.y );
#else
    ivec3 baseCoord = ivec3( ivec2(inPos.xy) * 2, 0 );
    out0 = ScreenSpaceToViewSpaceDepth( g_DepthSource.Load( baseCoord, ivec2( 0, 0 ) ).x );
    out1 = ScreenSpaceToViewSpaceDepth( g_DepthSource.Load( baseCoord, ivec2( 1, 0 ) ).x );
    out2 = ScreenSpaceToViewSpaceDepth( g_DepthSource.Load( baseCoord, ivec2( 0, 1 ) ).x );
    out3 = ScreenSpaceToViewSpaceDepth( g_DepthSource.Load( baseCoord, ivec2( 1, 1 ) ).x );
#endif

    float pixZs[4][4];

    // middle 4
    pixZs[1][1] = out0;
    pixZs[2][1] = out1;
    pixZs[1][2] = out2;
    pixZs[2][2] = out3;
    // left 2
    pixZs[0][1] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2( -1, 0 ) ).x );
    pixZs[0][2] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2( -1, 1 ) ).x );
    // right 2
    pixZs[3][1] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  2, 0 ) ).x );
    pixZs[3][2] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  2, 1 ) ).x );
    // top 2
    pixZs[1][0] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  0, -1 ) ).x );
    pixZs[2][0] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  1, -1 ) ).x );
    // bottom 2
    pixZs[1][3] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  0,  2 ) ).x );
    pixZs[2][3] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  1,  2 ) ).x );

    vec4 edges0 = CalculateEdges( pixZs[1][1], pixZs[0][1], pixZs[2][1], pixZs[1][0], pixZs[1][2] );
    vec4 edges1 = CalculateEdges( pixZs[2][1], pixZs[1][1], pixZs[3][1], pixZs[2][0], pixZs[2][2] );
    vec4 edges2 = CalculateEdges( pixZs[1][2], pixZs[0][2], pixZs[2][2], pixZs[1][1], pixZs[1][3] );
    vec4 edges3 = CalculateEdges( pixZs[2][2], pixZs[1][2], pixZs[3][2], pixZs[2][1], pixZs[2][3] );

    vec3 pixPos[4][4];
    // middle 4
    pixPos[1][1] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0,  0.0 ), pixZs[1][1] );
    pixPos[2][1] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0,  0.0 ), pixZs[2][1] );
    pixPos[1][2] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0,  1.0 ), pixZs[1][2] );
    pixPos[2][2] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0,  1.0 ), pixZs[2][2] );
    // left 2
    pixPos[0][1] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( -1.0,  0.0), pixZs[0][1] );
    pixPos[0][2] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( -1.0,  1.0), pixZs[0][2] );
    // right 2
    pixPos[3][1] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2(  2.0,  0.0), pixZs[3][1] );
    pixPos[3][2] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2(  2.0,  1.0), pixZs[3][2] );
    // top 2
    pixPos[1][0] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0, -1.0 ), pixZs[1][0] );
    pixPos[2][0] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0, -1.0 ), pixZs[2][0] );
    // bottom 2
    pixPos[1][3] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0,  2.0 ), pixZs[1][3] );
    pixPos[2][3] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0,  2.0 ), pixZs[2][3] );

    vec3 norm0 = CalculateNormal( edges0, pixPos[1][1], pixPos[0][1], pixPos[2][1], pixPos[1][0], pixPos[1][2] );
    vec3 norm1 = CalculateNormal( edges1, pixPos[2][1], pixPos[1][1], pixPos[3][1], pixPos[2][0], pixPos[2][2] );
    vec3 norm2 = CalculateNormal( edges2, pixPos[1][2], pixPos[0][2], pixPos[2][2], pixPos[1][1], pixPos[1][3] );
    vec3 norm3 = CalculateNormal( edges3, pixPos[2][2], pixPos[1][2], pixPos[3][2], pixPos[2][1], pixPos[2][3] );

    g_NormalsOutputUAV[ baseCoords + ivec2( 0, 0 ) ] = vec4( norm0 * 0.5 + 0.5, 0.0 );
    g_NormalsOutputUAV[ baseCoords + ivec2( 1, 0 ) ] = vec4( norm1 * 0.5 + 0.5, 0.0 );
    g_NormalsOutputUAV[ baseCoords + ivec2( 0, 1 ) ] = vec4( norm2 * 0.5 + 0.5, 0.0 );
    g_NormalsOutputUAV[ baseCoords + ivec2( 1, 1 ) ] = vec4( norm3 * 0.5 + 0.5, 0.0 );
}

void PSPrepareDepthsAndNormalsHalf( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1 )
{
    ivec2 baseCoords = (( ivec2 )inPos.xy) * 2;
    vec2 upperLeftUV = (inPos.xy - 0.25) * g_ASSAOConsts.Viewport2xPixelSize;

    ivec3 baseCoord = ivec3( ivec2(inPos.xy) * 2, 0 );
    float z0 = ScreenSpaceToViewSpaceDepth( g_DepthSource.Load( baseCoord, ivec2( 0, 0 ) ).x );
    float z1 = ScreenSpaceToViewSpaceDepth( g_DepthSource.Load( baseCoord, ivec2( 1, 0 ) ).x );
    float z2 = ScreenSpaceToViewSpaceDepth( g_DepthSource.Load( baseCoord, ivec2( 0, 1 ) ).x );
    float z3 = ScreenSpaceToViewSpaceDepth( g_DepthSource.Load( baseCoord, ivec2( 1, 1 ) ).x );

    out0 = z0;
    out1 = z3;

    float pixZs[4][4];

    // middle 4
    pixZs[1][1] = z0;
    pixZs[2][1] = z1;
    pixZs[1][2] = z2;
    pixZs[2][2] = z3;
    // left 2
    pixZs[0][1] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2( -1, 0 ) ).x );
    pixZs[0][2] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2( -1, 1 ) ).x );
    // right 2
    pixZs[3][1] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  2, 0 ) ).x );
    pixZs[3][2] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  2, 1 ) ).x );
    // top 2
    pixZs[1][0] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  0, -1 ) ).x );
    pixZs[2][0] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  1, -1 ) ).x );
    // bottom 2
    pixZs[1][3] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  0,  2 ) ).x );
    pixZs[2][3] = ScreenSpaceToViewSpaceDepth(  g_DepthSource.SampleLevel( g_PointClampSampler, upperLeftUV, 0.0, ivec2(  1,  2 ) ).x );

    vec4 edges0 = CalculateEdges( pixZs[1][1], pixZs[0][1], pixZs[2][1], pixZs[1][0], pixZs[1][2] );
    vec4 edges1 = CalculateEdges( pixZs[2][1], pixZs[1][1], pixZs[3][1], pixZs[2][0], pixZs[2][2] );
    vec4 edges2 = CalculateEdges( pixZs[1][2], pixZs[0][2], pixZs[2][2], pixZs[1][1], pixZs[1][3] );
    vec4 edges3 = CalculateEdges( pixZs[2][2], pixZs[1][2], pixZs[3][2], pixZs[2][1], pixZs[2][3] );

    vec3 pixPos[4][4];

    // there is probably a way to optimize the math below; however no approximation will work, has to be precise.

    // middle 4
    pixPos[1][1] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0,  0.0 ), pixZs[1][1] );
    pixPos[2][1] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0,  0.0 ), pixZs[2][1] );
    pixPos[1][2] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0,  1.0 ), pixZs[1][2] );
    pixPos[2][2] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0,  1.0 ), pixZs[2][2] );
    // left 2
    pixPos[0][1] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( -1.0,  0.0), pixZs[0][1] );
    //pixPos[0][2] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( -1.0,  1.0), pixZs[0][2] );
    // right 2
    //pixPos[3][1] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2(  2.0,  0.0), pixZs[3][1] );
    pixPos[3][2] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2(  2.0,  1.0), pixZs[3][2] );
    // top 2
    pixPos[1][0] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0, -1.0 ), pixZs[1][0] );
    //pixPos[2][0] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0, -1.0 ), pixZs[2][0] );
    // bottom 2
    //pixPos[1][3] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0,  2.0 ), pixZs[1][3] );
    pixPos[2][3] = NDCToViewspace( upperLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0,  2.0 ), pixZs[2][3] );

    vec3 norm0 = CalculateNormal( edges0, pixPos[1][1], pixPos[0][1], pixPos[2][1], pixPos[1][0], pixPos[1][2] );
    vec3 norm3 = CalculateNormal( edges3, pixPos[2][2], pixPos[1][2], pixPos[3][2], pixPos[2][1], pixPos[2][3] );

    g_NormalsOutputUAV[ baseCoords + ivec2( 0, 0 ) ] = vec4( norm0 * 0.5 + 0.5, 0.0 );
    g_NormalsOutputUAV[ baseCoords + ivec2( 1, 1 ) ] = vec4( norm3 * 0.5 + 0.5, 0.0 );
}

void PrepareDepthMip( const vec4 inPos/*, const vec2 inUV*/, int mipLevel, out float outD0, out float outD1, out float outD2, out float outD3 )
{
    ivec2 baseCoords = ivec2(inPos.xy) * 2;

    vec4 depthsArr[4];
    float depthsOutArr[4];

    // how to Gather a specific mip level?
    depthsArr[0].x = g_ViewspaceDepthSource[baseCoords + ivec2( 0, 0 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[0].y = g_ViewspaceDepthSource[baseCoords + ivec2( 1, 0 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[0].z = g_ViewspaceDepthSource[baseCoords + ivec2( 0, 1 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[0].w = g_ViewspaceDepthSource[baseCoords + ivec2( 1, 1 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].x = g_ViewspaceDepthSource1[baseCoords + ivec2( 0, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].y = g_ViewspaceDepthSource1[baseCoords + ivec2( 1, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].z = g_ViewspaceDepthSource1[baseCoords + ivec2( 0, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].w = g_ViewspaceDepthSource1[baseCoords + ivec2( 1, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].x = g_ViewspaceDepthSource2[baseCoords + ivec2( 0, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].y = g_ViewspaceDepthSource2[baseCoords + ivec2( 1, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].z = g_ViewspaceDepthSource2[baseCoords + ivec2( 0, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].w = g_ViewspaceDepthSource2[baseCoords + ivec2( 1, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].x = g_ViewspaceDepthSource3[baseCoords + ivec2( 0, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].y = g_ViewspaceDepthSource3[baseCoords + ivec2( 1, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].z = g_ViewspaceDepthSource3[baseCoords + ivec2( 0, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].w = g_ViewspaceDepthSource3[baseCoords + ivec2( 1, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;

    const uvec2 SVPosui         = uvec2( inPos.xy );
    const uint pseudoRandomA    = (SVPosui.x ) + 2 * (SVPosui.y );

    float dummyUnused1;
    float dummyUnused2;
    float falloffCalcMulSq, falloffCalcAdd;

    [unroll]
    for( int i = 0; i < 4; i++ )
    {
        vec4 depths = depthsArr[i];

        float closest = min( min( depths.x, depths.y ), min( depths.z, depths.w ) );

        CalculateRadiusParameters( abs( closest ), 1.0, dummyUnused1, dummyUnused2, falloffCalcMulSq );

        vec4 dists = depths - closest.xxxx;

        vec4 weights = clamp( dists * dists * falloffCalcMulSq + 1.0, 0.0, 1.0 );

        float smartAvg = dot( weights, depths ) / dot( weights, vec4( 1.0, 1.0, 1.0, 1.0 ) );

        const uint pseudoRandomIndex = ( pseudoRandomA + i ) % 4;

        //depthsOutArr[i] = closest;
        //depthsOutArr[i] = depths[ pseudoRandomIndex ];
        depthsOutArr[i] = smartAvg;
    }

    outD0 = depthsOutArr[0];
    outD1 = depthsOutArr[1];
    outD2 = depthsOutArr[2];
    outD3 = depthsOutArr[3];
}

void PSPrepareDepthMip1( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
{
    PrepareDepthMip( inPos/*, inUV*/, 1, out0, out1, out2, out3 );
}

void PSPrepareDepthMip2( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
{
    PrepareDepthMip( inPos/*, inUV*/, 2, out0, out1, out2, out3 );
}

void PSPrepareDepthMip3( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
{
    PrepareDepthMip( inPos/*, inUV*/, 3, out0, out1, out2, out3 );
}

vec3 DecodeNormal( vec3 encodedNormal )
{
    vec3 normal = encodedNormal * g_ASSAOConsts.NormalsUnpackMul.xxx + g_ASSAOConsts.NormalsUnpackAdd.xxx;

#if SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
    normal = mul( normal, (mat3)g_ASSAOConsts.NormalsWorldToViewspaceMatrix ).xyz;
#endif

    // normal = normalize( normal );    // normalize adds around 2.5% cost on High settings but makes little (PSNR 66.7) visual difference when normals are as in the sample (stored in R8G8B8A8_UNORM,
    //                                  // decoded in the shader), however it will likely be required if using different encoding/decoding or the inputs are not normalized, etc.

    return normal;
}

vec3 LoadNormal( ivec2 pos )
{
    vec3 encodedNormal = g_NormalmapSource.Load( ivec3( pos, 0 ) ).xyz;
    return DecodeNormal( encodedNormal );
}

vec3 LoadNormal( ivec2 pos, ivec2 offset )
{
    vec3 encodedNormal = g_NormalmapSource.Load( ivec3( pos, 0 ), offset ).xyz;
    return DecodeNormal( encodedNormal );
}

// all vectors in viewspace
float CalculatePixelObscurance( vec3 pixelNormal, vec3 hitDelta, float falloffCalcMulSq )
{
  float lengthSq = dot( hitDelta, hitDelta );
  float NdotD = dot(pixelNormal, hitDelta) / sqrt(lengthSq);

  float falloffMult = max( 0.0, lengthSq * falloffCalcMulSq + 1.0 );

  return max( 0, NdotD - g_ASSAOConsts.EffectHorizonAngleThreshold ) * falloffMult;
}

void SSAOTapInner( const int qualityLevel, inout float obscuranceSum, inout float weightSum, const vec2 samplingUV, const float mipLevel, const vec3 pixCenterPos, const vec3 negViewspaceDir,vec3 pixelNormal, const float falloffCalcMulSq, const float weightMod, const int dbgTapIndex )
{
    // get depth at sample
    float viewspaceSampleZ = g_ViewspaceDepthSource.SampleLevel( g_ViewspaceDepthTapSampler, samplingUV.xy, mipLevel ).x; // * g_ASSAOConsts.MaxViewspaceDepth;

    // convert to viewspace
    vec3 hitPos = NDCToViewspace( samplingUV.xy, viewspaceSampleZ ).xyz;
    vec3 hitDelta = hitPos - pixCenterPos;

    float obscurance = CalculatePixelObscurance( pixelNormal, hitDelta, falloffCalcMulSq );
    float weight = 1.0;

    if( qualityLevel >= SSAO_HALOING_REDUCTION_ENABLE_AT_QUALITY_PRESET )
    {
        //float reduct = max( 0, dot( hitDelta, negViewspaceDir ) );
        float reduct = max( 0, -hitDelta.z ); // cheaper, less correct version
        reduct = clamp( reduct * g_ASSAOConsts.NegRecEffectRadius + 2.0, 0.0, 1.0 ); // saturate( 2.0 - reduct / g_ASSAOConsts.EffectRadius );
        weight = SSAO_HALOING_REDUCTION_AMOUNT * reduct + (1.0 - SSAO_HALOING_REDUCTION_AMOUNT);
    }
    weight *= weightMod;
    obscuranceSum += obscurance * weight;
    weightSum += weight;
}

void SSAOTap( const int qualityLevel, inout float obscuranceSum, inout float weightSum, const int tapIndex, const mat2 rotScale, const vec3 pixCenterPos, const vec3 negViewspaceDir, vec3 pixelNormal, const vec2 normalizedScreenPos, const float mipOffset, const float falloffCalcMulSq, float weightMod, vec2 normXY, float normXYLength )
{
    vec2  sampleOffset;
    float   samplePow2Len;

    // patterns
    {
        vec4 newSample = g_samplePatternMain[tapIndex];
        sampleOffset    = mul( rotScale, newSample.xy );
        samplePow2Len   = newSample.w;                      // precalculated, same as: samplePow2Len = log2( length( newSample.xy ) );
        weightMod *= newSample.z;
    }

    // snap to pixel center (more correct obscurance math, avoids artifacts)
    sampleOffset                    = round(sampleOffset);

    // calculate MIP based on the sample distance from the centre, similar to as described
    // in http://graphics.cs.williams.edu/papers/SAOHPG12/.
    float mipLevel = ( qualityLevel < SSAO_DEPTH_MIPS_ENABLE_AT_QUALITY_PRESET )?(0):(samplePow2Len + mipOffset);

    vec2 samplingUV = sampleOffset * g_ASSAOConsts.Viewport2xPixelSize + normalizedScreenPos;

    SSAOTapInner( qualityLevel, obscuranceSum, weightSum, samplingUV, mipLevel, pixCenterPos, negViewspaceDir, pixelNormal, falloffCalcMulSq, weightMod, tapIndex * 2 );

    // for the second tap, just use the mirrored offset
    vec2 sampleOffsetMirroredUV    = -sampleOffset;

    // tilt the second set of samples so that the disk is effectively rotated by the normal
    // effective at removing one set of artifacts, but too expensive for lower quality settings
    if( qualityLevel >= SSAO_TILT_SAMPLES_ENABLE_AT_QUALITY_PRESET )
    {
        float dotNorm = dot( sampleOffsetMirroredUV, normXY );
        sampleOffsetMirroredUV -= dotNorm * normXYLength * normXY;
        sampleOffsetMirroredUV = round(sampleOffsetMirroredUV);
    }

    // snap to pixel center (more correct obscurance math, avoids artifacts)
    vec2 samplingMirroredUV = sampleOffsetMirroredUV * g_ASSAOConsts.Viewport2xPixelSize + normalizedScreenPos;

    SSAOTapInner( qualityLevel, obscuranceSum, weightSum, samplingMirroredUV, mipLevel, pixCenterPos, negViewspaceDir, pixelNormal, falloffCalcMulSq, weightMod, tapIndex * 2 + 1 );
}

// this function is designed to only work with half/half depth at the moment - there's a couple of hardcoded paths that expect pixel/texel size, so it will not work for full res
void GenerateSSAOShadowsInternal( out float outShadowTerm, out vec4 outEdges, out float outWeight, const vec2 SVPos/*, const vec2 normalizedScreenPos*/, uniform int qualityLevel, bool adaptiveBase )
{
    vec2 SVPosRounded = trunc( SVPos );
    uvec2 SVPosui = uvec2( SVPosRounded ); //same as uvec2( SVPos )

    const int numberOfTaps = (adaptiveBase)?(SSAO_ADAPTIVE_TAP_BASE_COUNT) : ( g_numTaps[qualityLevel] );
    float pixZ, pixLZ, pixTZ, pixRZ, pixBZ;

    vec4 valuesUL     = g_ViewspaceDepthSource.GatherRed( g_PointMirrorSampler, SVPosRounded * g_ASSAOConsts.HalfViewportPixelSize );
    vec4 valuesBR     = g_ViewspaceDepthSource.GatherRed( g_PointMirrorSampler, SVPosRounded * g_ASSAOConsts.HalfViewportPixelSize, ivec2( 1, 1 ) );

    // get this pixel's viewspace depth
    pixZ = valuesUL.y; //float pixZ = g_ViewspaceDepthSource.SampleLevel( g_PointMirrorSampler, normalizedScreenPos, 0.0 ).x; // * g_ASSAOConsts.MaxViewspaceDepth;

    // get left right top bottom neighbouring pixels for edge detection (gets compiled out on qualityLevel == 0)
    pixLZ   = valuesUL.x;
    pixTZ   = valuesUL.z;
    pixRZ   = valuesBR.z;
    pixBZ   = valuesBR.x;

    vec2 normalizedScreenPos = SVPosRounded * g_ASSAOConsts.Viewport2xPixelSize + g_ASSAOConsts.Viewport2xPixelSize_x_025;
    vec3 pixCenterPos = NDCToViewspace( normalizedScreenPos, pixZ ); // g

    // Load this pixel's viewspace normal
    uvec2 fullResCoord = SVPosui * 2 + g_ASSAOConsts.PerPassFullResCoordOffset.xy;
    vec3 pixelNormal = LoadNormal( fullResCoord );

    const vec2 pixelDirRBViewspaceSizeAtCenterZ = pixCenterPos.z * g_ASSAOConsts.NDCToViewMul * g_ASSAOConsts.Viewport2xPixelSize;  // optimized approximation of:  vec2 pixelDirRBViewspaceSizeAtCenterZ = NDCToViewspace( normalizedScreenPos.xy + g_ASSAOConsts.ViewportPixelSize.xy, pixCenterPos.z ).xy - pixCenterPos.xy;

    float pixLookupRadiusMod;
    float falloffCalcMulSq;

    // calculate effect radius and fit our screen sampling pattern inside it
    float effectViewspaceRadius;
    CalculateRadiusParameters( length( pixCenterPos ), pixelDirRBViewspaceSizeAtCenterZ, pixLookupRadiusMod, effectViewspaceRadius, falloffCalcMulSq );

    // calculate samples rotation/scaling
    mat2 rotScale;
    {
        // reduce effect radius near the screen edges slightly; ideally, one would render a larger depth buffer (5% on each side) instead
        if( !adaptiveBase && (qualityLevel >= SSAO_REDUCE_RADIUS_NEAR_SCREEN_BORDER_ENABLE_AT_QUALITY_PRESET) )
        {
            float nearScreenBorder = min( min( normalizedScreenPos.x, 1.0 - normalizedScreenPos.x ), min( normalizedScreenPos.y, 1.0 - normalizedScreenPos.y ) );
            nearScreenBorder = clamp( 10.0 * nearScreenBorder + 0.6, 0.0, 1.0 );
            pixLookupRadiusMod *= nearScreenBorder;
        }

        // load & update pseudo-random rotation matrix
        uint pseudoRandomIndex = uint( SVPosRounded.y * 2 + SVPosRounded.x ) % 5;
        vec4 rs = g_ASSAOConsts.PatternRotScaleMatrices[ pseudoRandomIndex ];
        rotScale = mat2( rs.x * pixLookupRadiusMod, rs.y * pixLookupRadiusMod, rs.z * pixLookupRadiusMod, rs.w * pixLookupRadiusMod );
    }

    // the main obscurance & sample weight storage
    float obscuranceSum = 0.0;
    float weightSum = 0.0;

    // edge mask for between this and left/right/top/bottom neighbour pixels - not used in quality level 0 so initialize to "no edge" (1 is no edge, 0 is edge)
    vec4 edgesLRTB = vec4( 1.0, 1.0, 1.0, 1.0 );

    // Move center pixel slightly towards camera to avoid imprecision artifacts due to using of 16bit depth buffer; a lot smaller offsets needed when using 32bit floats
    pixCenterPos *= g_ASSAOConsts.DepthPrecisionOffsetMod;

    if( !adaptiveBase && (qualityLevel >= SSAO_DEPTH_BASED_EDGES_ENABLE_AT_QUALITY_PRESET) )
    {
        edgesLRTB = CalculateEdges( pixZ, pixLZ, pixRZ, pixTZ, pixBZ );
    }

    // adds a more high definition sharp effect, which gets blurred out (reuses left/right/top/bottom samples that we used for edge detection)
    if( !adaptiveBase && (qualityLevel >= SSAO_DETAIL_AO_ENABLE_AT_QUALITY_PRESET) )
    {
        // disable in case of quality level 4 (reference)
        if( qualityLevel != 4 )
        {
            //approximate neighbouring pixels positions (actually just deltas or "positions - pixCenterPos" )
            vec3 viewspaceDirZNormalized = vec3( pixCenterPos.xy / pixCenterPos.zz, 1.0 );
            vec3 pixLDelta  = vec3( -pixelDirRBViewspaceSizeAtCenterZ.x, 0.0, 0.0 ) + viewspaceDirZNormalized * (pixLZ - pixCenterPos.z); // very close approximation of: vec3 pixLPos  = NDCToViewspace( normalizedScreenPos + vec2( -g_ASSAOConsts.HalfViewportPixelSize.x, 0.0 ), pixLZ ).xyz - pixCenterPos.xyz;
            vec3 pixRDelta  = vec3( +pixelDirRBViewspaceSizeAtCenterZ.x, 0.0, 0.0 ) + viewspaceDirZNormalized * (pixRZ - pixCenterPos.z); // very close approximation of: vec3 pixRPos  = NDCToViewspace( normalizedScreenPos + vec2( +g_ASSAOConsts.HalfViewportPixelSize.x, 0.0 ), pixRZ ).xyz - pixCenterPos.xyz;
            vec3 pixTDelta  = vec3( 0.0, -pixelDirRBViewspaceSizeAtCenterZ.y, 0.0 ) + viewspaceDirZNormalized * (pixTZ - pixCenterPos.z); // very close approximation of: vec3 pixTPos  = NDCToViewspace( normalizedScreenPos + vec2( 0.0, -g_ASSAOConsts.HalfViewportPixelSize.y ), pixTZ ).xyz - pixCenterPos.xyz;
            vec3 pixBDelta  = vec3( 0.0, +pixelDirRBViewspaceSizeAtCenterZ.y, 0.0 ) + viewspaceDirZNormalized * (pixBZ - pixCenterPos.z); // very close approximation of: vec3 pixBPos  = NDCToViewspace( normalizedScreenPos + vec2( 0.0, +g_ASSAOConsts.HalfViewportPixelSize.y ), pixBZ ).xyz - pixCenterPos.xyz;

            const float rangeReductionConst         = 4.0f;                         // this is to avoid various artifacts
            const float modifiedFalloffCalcMulSq    = rangeReductionConst * falloffCalcMulSq;

            vec4 additionalObscurance;
            additionalObscurance.x = CalculatePixelObscurance( pixelNormal, pixLDelta, modifiedFalloffCalcMulSq );
            additionalObscurance.y = CalculatePixelObscurance( pixelNormal, pixRDelta, modifiedFalloffCalcMulSq );
            additionalObscurance.z = CalculatePixelObscurance( pixelNormal, pixTDelta, modifiedFalloffCalcMulSq );
            additionalObscurance.w = CalculatePixelObscurance( pixelNormal, pixBDelta, modifiedFalloffCalcMulSq );

            obscuranceSum += g_ASSAOConsts.DetailAOStrength * dot( additionalObscurance, edgesLRTB );
        }
    }

    // Sharp normals also create edges - but this adds to the cost as well
    if( !adaptiveBase && (qualityLevel >= SSAO_NORMAL_BASED_EDGES_ENABLE_AT_QUALITY_PRESET ) )
    {
        vec3 neighbourNormalL  = LoadNormal( fullResCoord, ivec2( -2,  0 ) );
        vec3 neighbourNormalR  = LoadNormal( fullResCoord, ivec2(  2,  0 ) );
        vec3 neighbourNormalT  = LoadNormal( fullResCoord, ivec2(  0, -2 ) );
        vec3 neighbourNormalB  = LoadNormal( fullResCoord, ivec2(  0,  2 ) );

        const float dotThreshold = SSAO_NORMAL_BASED_EDGES_DOT_THRESHOLD;

        vec4 normalEdgesLRTB;
        normalEdgesLRTB.x = clamp( (dot( pixelNormal, neighbourNormalL ) + dotThreshold ), 0.0, 1.0 );
        normalEdgesLRTB.y = clamp( (dot( pixelNormal, neighbourNormalR ) + dotThreshold ), 0.0, 1.0 );
        normalEdgesLRTB.z = clamp( (dot( pixelNormal, neighbourNormalT ) + dotThreshold ), 0.0, 1.0 );
        normalEdgesLRTB.w = clamp( (dot( pixelNormal, neighbourNormalB ) + dotThreshold ), 0.0, 1.0 );

//#define SSAO_SMOOTHEN_NORMALS // fixes some aliasing artifacts but kills a lot of high detail and adds to the cost - not worth it probably but feel free to play with it
#ifdef SSAO_SMOOTHEN_NORMALS
        //neighbourNormalL  = LoadNormal( fullResCoord, ivec2( -1,  0 ) );
        //neighbourNormalR  = LoadNormal( fullResCoord, ivec2(  1,  0 ) );
        //neighbourNormalT  = LoadNormal( fullResCoord, ivec2(  0, -1 ) );
        //neighbourNormalB  = LoadNormal( fullResCoord, ivec2(  0,  1 ) );
        pixelNormal += neighbourNormalL * edgesLRTB.x + neighbourNormalR * edgesLRTB.y + neighbourNormalT * edgesLRTB.z + neighbourNormalB * edgesLRTB.w;
        pixelNormal = normalize( pixelNormal );
#endif

        edgesLRTB *= normalEdgesLRTB;
    }



    const float globalMipOffset     = SSAO_DEPTH_MIPS_GLOBAL_OFFSET;
    float mipOffset = ( qualityLevel < SSAO_DEPTH_MIPS_ENABLE_AT_QUALITY_PRESET ) ? ( 0 ) : ( log2( pixLookupRadiusMod ) + globalMipOffset );

    // Used to tilt the second set of samples so that the disk is effectively rotated by the normal
    // effective at removing one set of artifacts, but too expensive for lower quality settings
    vec2 normXY = vec2( pixelNormal.x, pixelNormal.y );
    float normXYLength = length( normXY );
    normXY /= vec2( normXYLength, -normXYLength );
    normXYLength *= SSAO_TILT_SAMPLES_AMOUNT;

    const vec3 negViewspaceDir = -normalize( pixCenterPos );

    // standard, non-adaptive approach
    if( (qualityLevel != 3) || adaptiveBase )
    {
        // [unroll] // <- doesn't seem to help on any platform, although the compilers seem to unroll anyway if const number of tap used!
        for( int i = 0; i < numberOfTaps; i++ )
        {
            SSAOTap( qualityLevel, obscuranceSum, weightSum, i, rotScale, pixCenterPos, negViewspaceDir, pixelNormal, normalizedScreenPos, mipOffset, falloffCalcMulSq, 1.0, normXY, normXYLength );
        }
    }

    // calculate weighted average
    float obscurance = obscuranceSum / weightSum;

    // calculate fadeout (1 close, gradient, 0 far)
    float fadeOut = clamp( pixCenterPos.z * g_ASSAOConsts.EffectFadeOutMul + g_ASSAOConsts.EffectFadeOutAdd, 0.0, 1.0 );

    // Reduce the SSAO shadowing if we're on the edge to remove artifacts on edges (we don't care for the lower quality one)
    if( !adaptiveBase && (qualityLevel >= SSAO_DEPTH_BASED_EDGES_ENABLE_AT_QUALITY_PRESET) )
    {
        // float edgeCount = dot( 1.0-edgesLRTB, vec4( 1.0, 1.0, 1.0, 1.0 ) );

        // when there's more than 2 opposite edges, start fading out the occlusion to reduce aliasing artifacts
        float edgeFadeoutFactor = clamp( (1.0 - edgesLRTB.x - edgesLRTB.y) * 0.35, 0.0, 1.0) + clamp( (1.0 - edgesLRTB.z - edgesLRTB.w) * 0.35, 0.0, 1.0 );

        // (experimental) if you want to reduce the effect next to any edge
        // edgeFadeoutFactor += 0.1 * clamp( dot( 1 - edgesLRTB, vec4( 1, 1, 1, 1 ) ) );

        fadeOut *= clamp( 1.0 - edgeFadeoutFactor, 0.0, 1.0 );
    }

    // same as a bove, but a lot more conservative version
    // fadeOut *= saturate( dot( edgesLRTB, vec4( 0.9, 0.9, 0.9, 0.9 ) ) - 2.6 );

    // strength
    obscurance = g_ASSAOConsts.EffectShadowStrength * obscurance;

    // clamp
    obscurance = min( obscurance, g_ASSAOConsts.EffectShadowClamp );

    // fadeout
    obscurance *= fadeOut;

    // conceptually switch to occlusion with the meaning being visibility (grows with visibility, occlusion == 1 implies full visibility),
    // to be in line with what is more commonly used.
    float occlusion = 1.0 - obscurance;

    // modify the gradient
    // note: this cannot be moved to a later pass because of loss of precision after storing in the render target
    occlusion = pow( clamp( occlusion, 0.0, 1.0 ), g_ASSAOConsts.EffectShadowPow );

    // outputs!
    outShadowTerm   = occlusion;    // Our final 'occlusion' term (0 means fully occluded, 1 means fully lit)
    outEdges        = edgesLRTB;    // These are used to prevent blurring across edges, 1 means no edge, 0 means edge, 0.5 means half way there, etc.
    outWeight       = weightSum;
}

void PSGenerateQ0( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out vec2 out0 : SV_Target0 )
{
    float   outShadowTerm;
    float   outWeight;
    vec4  outEdges;
    GenerateSSAOShadowsInternal( outShadowTerm, outEdges, outWeight, inPos.xy/*, inUV*/, 0, false );
    out0.x = outShadowTerm;
    out0.y = PackEdges( vec4( 1, 1, 1, 1 ) ); // no edges in low quality
}

// Qi used for quality preset i (low, medium, high, highest)
void PSGenerateQ1( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out vec2 out0 : SV_Target0 )
{
    float   outShadowTerm;
    float   outWeight;
    vec4  outEdges;
    GenerateSSAOShadowsInternal( outShadowTerm, outEdges, outWeight, inPos.xy/*, inUV*/, 1, false );
    out0.x = outShadowTerm;
    out0.y = PackEdges( outEdges );
}

void PSGenerateQ2( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out vec2 out0 : SV_Target0 )
{
    float   outShadowTerm;
    float   outWeight;
    vec4  outEdges;
    GenerateSSAOShadowsInternal( outShadowTerm, outEdges, outWeight, inPos.xy/*, inUV*/, 2, false );
    out0.x = outShadowTerm;
    out0.y = PackEdges( outEdges );
}

void PSGenerateQ3( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out vec2 out0 : SV_Target0 )
{
    float   outShadowTerm;
    float   outWeight;
    vec4  outEdges;
    GenerateSSAOShadowsInternal( outShadowTerm, outEdges, outWeight, inPos.xy/*, inUV*/, 3, false);
    out0.x = outShadowTerm;
    out0.y = PackEdges( outEdges );
}

// ********************************************************************************************************
// Pixel shader that does smart blurring (to avoid bleeding)

void AddSample( float ssaoValue, float edgeValue, inout float sum, inout float sumWeight )
{
    float weight = edgeValue;

    sum += (weight * ssaoValue);
    sumWeight += weight;
}

vec2 SampleBlurredWide( vec4 inPos, vec2 coord )
{
    vec2 vC           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2( 0,  0 ) ).xy;
    vec2 vL           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2( -2, 0 ) ).xy;
    vec2 vT           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2( 0, -2 ) ).xy;
    vec2 vR           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2(  2, 0 ) ).xy;
    vec2 vB           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2( 0,  2 ) ).xy;

    float packedEdges   = vC.y;
    vec4 edgesLRTB    = UnpackEdges( packedEdges );
    edgesLRTB.x         *= UnpackEdges( vL.y ).y;
    edgesLRTB.z         *= UnpackEdges( vT.y ).w;
    edgesLRTB.y         *= UnpackEdges( vR.y ).x;
    edgesLRTB.w         *= UnpackEdges( vB.y ).z;

    float ssaoValue     = vC.x;
    float ssaoValueL    = vL.x;
    float ssaoValueT    = vT.x;
    float ssaoValueR    = vR.x;
    float ssaoValueB    = vB.x;

    float sumWeight = 0.8f;
    float sum = ssaoValue * sumWeight;

    AddSample( ssaoValueL, edgesLRTB.x, sum, sumWeight );
    AddSample( ssaoValueR, edgesLRTB.y, sum, sumWeight );
    AddSample( ssaoValueT, edgesLRTB.z, sum, sumWeight );
    AddSample( ssaoValueB, edgesLRTB.w, sum, sumWeight );

    float ssaoAvg = sum / sumWeight;

    ssaoValue = ssaoAvg; //min( ssaoValue, ssaoAvg ) * 0.2 + ssaoAvg * 0.8;

    return vec2( ssaoValue, packedEdges );
}

vec2 SampleBlurred( vec4 inPos, vec2 coord )
{
    float packedEdges   = g_BlurInput.Load( ivec3( inPos.xy, 0 ) ).y;
    vec4 edgesLRTB    = UnpackEdges( packedEdges );

                                                                                // automatically done in our shader
    vec4 valuesUL     = g_BlurInput.GatherRed( g_PointMirrorSampler, coord - g_ASSAOConsts.HalfViewportPixelSize * 0.5 );
    vec4 valuesBR     = g_BlurInput.GatherRed( g_PointMirrorSampler, coord + g_ASSAOConsts.HalfViewportPixelSize * 0.5 );

    // fetch all ssaoValues around current pixel
    float ssaoValue     = valuesUL.y;   // center   e.g. (5,5)                                          vUL.z
    float ssaoValueL    = valuesUL.x;   // left     --> (4,5)                                   vUL.x   vUL.y   vBR.z
    float ssaoValueT    = valuesUL.z;   // top      --> (5,4)                                           vBR.x   vBR.y
    float ssaoValueR    = valuesBR.z;   // right    valuesBR.z == (6,6) --> .z = (6,5)
    float ssaoValueB    = valuesBR.x;   // bottom   --> (5,6)

    float sumWeight = 0.5f;
    float sum = ssaoValue * sumWeight;

    AddSample( ssaoValueL, edgesLRTB.x, sum, sumWeight );
    AddSample( ssaoValueR, edgesLRTB.y, sum, sumWeight );

    AddSample( ssaoValueT, edgesLRTB.z, sum, sumWeight );
    AddSample( ssaoValueB, edgesLRTB.w, sum, sumWeight );

    float ssaoAvg = sum / sumWeight;

    ssaoValue = ssaoAvg; //min( ssaoValue, ssaoAvg ) * 0.2 + ssaoAvg * 0.8;

    return vec2( ssaoValue, packedEdges );
}

// edge-sensitive blur
vec2 PSSmartBlur( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    return SampleBlurred( inPos, inUV );
}

// edge-sensitive blur (wider kernel)
vec2 PSSmartBlurWide( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    return SampleBlurredWide( inPos, inUV );
}

vec4 PSApply( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/ ) : SV_Target
{
    float ao;
    uvec2 pixPos     = (uvec2)inPos.xy;
    uvec2 pixPosHalf = pixPos / uvec2(2, 2);

    // calculate index in the four deinterleaved source array texture
    int mx = (pixPos.x % 2);
    int my = (pixPos.y % 2);
    int ic = mx + my * 2;       // center index
    int ih = (1-mx) + my * 2;   // neighbouring, horizontal
    int iv = mx + (1-my) * 2;   // neighbouring, vertical
    int id = (1-mx) + (1-my)*2; // diagonal

    vec2 centerVal = g_FinalSSAO.Load( ivec4( pixPosHalf, ic, 0 ) ).xy;

    ao = centerVal.x;

#if 1   // change to 0 if you want to disable last pass high-res blur (for debugging purposes, etc.)
    vec4 edgesLRTB = UnpackEdges( centerVal.y );

    // return 1.0 - vec4( edgesLRTB.x, edgesLRTB.y * 0.5 + edgesLRTB.w * 0.5, edgesLRTB.z, 0.0 ); // debug show edges

    // convert index shifts to sampling offsets
    float fmx   = (float)mx;
    float fmy   = (float)my;

    // in case of an edge, push sampling offsets away from the edge (towards pixel center)
    float fmxe  = (edgesLRTB.y - edgesLRTB.x);
    float fmye  = (edgesLRTB.w - edgesLRTB.z);

    // calculate final sampling offsets and sample using bilinear filter
    vec2  uvH = (inPos.xy + vec2( fmx + fmxe - 0.5, 0.5 - fmy ) ) * 0.5 * g_ASSAOConsts.HalfViewportPixelSize;
    float   aoH = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( uvH, ih ), 0 ).x;
    vec2  uvV = (inPos.xy + vec2( 0.5 - fmx, fmy - 0.5 + fmye ) ) * 0.5 * g_ASSAOConsts.HalfViewportPixelSize;
    float   aoV = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( uvV, iv ), 0 ).x;
    vec2  uvD = (inPos.xy + vec2( fmx - 0.5 + fmxe, fmy - 0.5 + fmye ) ) * 0.5 * g_ASSAOConsts.HalfViewportPixelSize;
    float   aoD = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( uvD, id ), 0 ).x;

    // reduce weight for samples near edge - if the edge is on both sides, weight goes to 0
    vec4 blendWeights;
    blendWeights.x = 1.0;
    blendWeights.y = (edgesLRTB.x + edgesLRTB.y) * 0.5;
    blendWeights.z = (edgesLRTB.z + edgesLRTB.w) * 0.5;
    blendWeights.w = (blendWeights.y + blendWeights.z) * 0.5;

    // calculate weighted average
    float blendWeightsSum   = dot( blendWeights, vec4( 1.0, 1.0, 1.0, 1.0 ) );
    ao = dot( vec4( ao, aoH, aoV, aoD ), blendWeights ) / blendWeightsSum;
#endif

    return vec4( ao.xxx, 1.0 );
}

// edge-ignorant blur in x and y directions, 9 pixels touched (for the lowest quality level 0)
vec2 PSNonSmartBlur( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    vec2 halfPixel = g_ASSAOConsts.HalfViewportPixelSize * 0.5f;

    vec2 centre = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV, 0.0 ).xy;

    vec4 vals;
    vals.x = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV + vec2( -halfPixel.x * 3, -halfPixel.y ), 0.0 ).x;
    vals.y = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV + vec2( +halfPixel.x, -halfPixel.y * 3 ), 0.0 ).x;
    vals.z = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV + vec2( -halfPixel.x, +halfPixel.y * 3 ), 0.0 ).x;
    vals.w = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV + vec2( +halfPixel.x * 3, +halfPixel.y ), 0.0 ).x;

    return vec2(dot( vals, 0.2.xxxx ) + centre.x * 0.2, centre.y);
}

// edge-ignorant blur & apply (for the lowest quality level 0)
vec4 PSNonSmartApply( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    float a = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 0 ), 0.0 ).x;
    float b = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 1 ), 0.0 ).x;
    float c = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 2 ), 0.0 ).x;
    float d = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 3 ), 0.0 ).x;
    float avg = (a+b+c+d) * 0.25;
    return vec4( avg.xxx, 1.0 );
}

// edge-ignorant blur & apply, skipping half pixels in checkerboard pattern (for the Lowest quality level 0 and Settings::SkipHalfPixelsOnLowQualityLevel == true )
vec4 PSNonSmartHalfApply( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    float a = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 0 ), 0.0 ).x;
    float d = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 3 ), 0.0 ).x;
    float avg = (a+d) * 0.5;
    return vec4( avg.xxx, 1.0 );
}

// this function is designed to only work with half/half depth at the moment - there's a couple of hardcoded paths that expect pixel/texel size, so it will not work for full res
void GenerateSSAOShadowsInternal( out float outShadowTerm, out vec4 outEdges, out float outWeight, const vec2 SVPos/*, const vec2 normalizedScreenPos*/, /*uniform*/ int qualityLevel, bool adaptiveBase, out vec3 normal )
{
    vec2 SVPosRounded = trunc( SVPos );
    uvec2 SVPosui = uvec2( SVPosRounded ); //same as uvec2( SVPos )

    //const int numberOfTaps = (adaptiveBase) ? (SSAO_ADAPTIVE_TAP_BASE_COUNT) : ( g_numTaps[qualityLevel] );
    const int numberOfTaps = int(g_numTaps[qualityLevel]);
    float pixZ, pixLZ, pixTZ, pixRZ, pixBZ;

#ifdef true
    vec4 valuesBL = textureGather(g_ViewspaceDepthSource, SVPosRounded * g_ASSAOConsts.HalfViewportPixelSize );
    vec4 valuesUR = textureGatherOffset(g_ViewspaceDepthSource, SVPosRounded * g_ASSAOConsts.HalfViewportPixelSize, ivec2( 1, 1 ) );

    // TODO: is this correct? - TEST WITH DIFFERENT DIRECTION: FROM TOP LEFT TO BOTTOM RIGHT
    // see 'else'
    // get this pixel's viewspace depth
    pixZ = valuesBL.y; //float pixZ = g_ViewspaceDepthSource.SampleLevel( g_PointMirrorSampler, normalizedScreenPos, 0.0 ).x; // * g_ASSAOConsts.MaxViewspaceDepth;

    // get left right top bottom neighbouring pixels for edge detection (gets compiled out on qualityLevel == 0)
    pixLZ   = valuesBL.x;
    pixTZ   = valuesUR.x;
    pixRZ   = valuesUR.z;
    pixBZ   = valuesBL.z;
#else
    // going from top left to bottom right to mimic intels direct3d version
    vec4 valuesUL = textureGather(g_ViewspaceDepthSource, SVPosRounded * g_ASSAOConsts.HalfViewportPixelSize );
    vec4 valuesBR = textureGatherOffset(g_ViewspaceDepthSource, SVPosRounded * g_ASSAOConsts.HalfViewportPixelSize, ivec2( 1, -1 ) );
    pixZ = valuesUL.z; //float pixZ = g_ViewspaceDepthSource.SampleLevel( g_PointMirrorSampler, normalizedScreenPos, 0.0 ).x; // * g_ASSAOConsts.MaxViewspaceDepth;

    // get left right top bottom neighbouring pixels for edge detection (gets compiled out on qualityLevel == 0)
    pixLZ   = valuesUL.w;
    pixTZ   = valuesUL.y;
    pixRZ   = valuesBR.y;
    pixBZ   = valuesBR.w;
#endif

    vec2 normalizedScreenPos = SVPosRounded * g_ASSAOConsts.Viewport2xPixelSize + g_ASSAOConsts.Viewport2xPixelSize_x_025;
    // the assumption that normalizedScreenPos is in NDC might be wrong
    // need to transform to ndc first!
    vec3 pixCenterPos = NDCToViewspace( normalizedScreenPos, pixZ ); // g

    // Load this pixel's viewspace normal
    uvec2 fullResCoord = SVPosui * 2 + g_ASSAOConsts.PerPassFullResCoordOffset.xy;
    vec3 pixelNormal = LoadNormal( ivec2(fullResCoord) );
	//vec3 pixelNormal = texelFetch(g_NormalmapSource, ivec2(fullResCoord), 0 ).xyz;
    //pixelNormal = transpose(inverse(mat3(g_ASSAOConsts.viewMX))) * (pixelNormal);
	normal = pixelNormal;
    //pixelNormal = -pixelNormal;

    // optimized approximation of:
    // vec2 pixelDirRBViewspaceSizeAtCenterZ = NDCToViewspace( normalizedScreenPos.xy + g_ASSAOConsts.ViewportPixelSize.xy, pixCenterPos.z ).xy - pixCenterPos.xy;
    //const vec2 pixelDirRBViewspaceSizeAtCenterZ = pixCenterPos.z * g_ASSAOConsts.NDCToViewMul * g_ASSAOConsts.Viewport2xPixelSize;
    // try with g_ASSAOConsts.ViewportPixelSize instead of g_ASSAOConsts.Viewport2xPixelSize
    const vec2 pixelDirRBViewspaceSizeAtCenterZ = (-pixCenterPos.z) * g_ASSAOConsts.NDCToViewMul * g_ASSAOConsts.ViewportPixelSize;

    // was between ~[-340, -520]
    // update: now at ~[215, 384]
    // intel is within ~[20, 192]
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
        // TODO: does this need to be transposed?
        // but opengl (column-major) and direct3d (row-major) have different matrix orders
        rotScale = mat2( rs.x * pixLookupRadiusMod, rs.y * pixLookupRadiusMod, rs.z * pixLookupRadiusMod, rs.w * pixLookupRadiusMod );
        //rotScale = mat2( rs.x * pixLookupRadiusMod, rs.z * pixLookupRadiusMod, rs.y * pixLookupRadiusMod, rs.w * pixLookupRadiusMod );
    }

    // the main obscurance & sample weight storage
    float obscuranceSum = 0.0;
    float weightSum = 0.0;
	float obstwo = 0.0;

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
            // TODO: approximation is not correct atm, re-do the math
            //vec3 pixLDelta  = vec3( -pixelDirRBViewspaceSizeAtCenterZ.x, 0.0, 0.0 ) + viewspaceDirZNormalized * (pixLZ - pixCenterPos.z);
            //vec3 pixRDelta  = vec3( +pixelDirRBViewspaceSizeAtCenterZ.x, 0.0, 0.0 ) + viewspaceDirZNormalized * (pixRZ - pixCenterPos.z);
            //vec3 pixTDelta  = vec3( 0.0, +pixelDirRBViewspaceSizeAtCenterZ.y, 0.0 ) + viewspaceDirZNormalized * (pixTZ - pixCenterPos.z);
            //vec3 pixBDelta  = vec3( 0.0, -pixelDirRBViewspaceSizeAtCenterZ.y, 0.0 ) + viewspaceDirZNormalized * (pixBZ - pixCenterPos.z);
            vec3 pixLDelta  = NDCToViewspace( normalizedScreenPos + vec2( -g_ASSAOConsts.HalfViewportPixelSize.x, 0.0 ), pixLZ ).xyz - pixCenterPos.xyz;
            vec3 pixRDelta  = NDCToViewspace( normalizedScreenPos + vec2( +g_ASSAOConsts.HalfViewportPixelSize.x, 0.0 ), pixRZ ).xyz - pixCenterPos.xyz;
            vec3 pixTDelta  = NDCToViewspace( normalizedScreenPos + vec2( 0.0, +g_ASSAOConsts.HalfViewportPixelSize.y ), pixTZ ).xyz - pixCenterPos.xyz;
            vec3 pixBDelta  = NDCToViewspace( normalizedScreenPos + vec2( 0.0, -g_ASSAOConsts.HalfViewportPixelSize.y ), pixBZ ).xyz - pixCenterPos.xyz;


            const float rangeReductionConst         = 4.0f;                         // this is to avoid various artifacts
            const float modifiedFalloffCalcMulSq    = rangeReductionConst * falloffCalcMulSq;

            vec4 additionalObscurance;
            additionalObscurance.x = CalculatePixelObscurance( pixelNormal, pixLDelta, modifiedFalloffCalcMulSq );
            additionalObscurance.y = CalculatePixelObscurance( pixelNormal, pixRDelta, modifiedFalloffCalcMulSq );
            additionalObscurance.z = CalculatePixelObscurance( pixelNormal, pixTDelta, modifiedFalloffCalcMulSq );
            additionalObscurance.w = CalculatePixelObscurance( pixelNormal, pixBDelta, modifiedFalloffCalcMulSq );

            obscuranceSum += g_ASSAOConsts.DetailAOStrength * dot( additionalObscurance, edgesLRTB );

			obstwo = pixLZ / 2.f;
        }
    }

    // Sharp normals also create edges - but this adds to the cost as well
    if( !adaptiveBase && (qualityLevel >= SSAO_NORMAL_BASED_EDGES_ENABLE_AT_QUALITY_PRESET ) )
    {
        // NOTE: this has to be in line with whichever gather is implemented around line ~26
        //vec3 neighbourNormalL = DecodeNormal( texelFetchOffset(g_NormalmapSource, ivec2(fullResCoord), 0, ivec2(-2,  0 ) ).xyz );
        //vec3 neighbourNormalR = DecodeNormal( texelFetchOffset(g_NormalmapSource, ivec2(fullResCoord), 0, ivec2( 2,  0 ) ).xyz );
        //vec3 neighbourNormalT = DecodeNormal( texelFetchOffset(g_NormalmapSource, ivec2(fullResCoord), 0, ivec2( 0,  2 ) ).xyz );
        //vec3 neighbourNormalB = DecodeNormal( texelFetchOffset(g_NormalmapSource, ivec2(fullResCoord), 0, ivec2( 0, -2 ) ).xyz );
        vec3 neighbourNormalL  = LoadNormal( ivec2(fullResCoord), ivec2( -2,  0 ) );
        vec3 neighbourNormalR  = LoadNormal( ivec2(fullResCoord), ivec2(  2,  0 ) );
        vec3 neighbourNormalT  = LoadNormal( ivec2(fullResCoord), ivec2(  0,  2 ) );
        vec3 neighbourNormalB  = LoadNormal( ivec2(fullResCoord), ivec2(  0, -2 ) );

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
    // else // if( qualityLevel == 3 ) adaptive approach
    // {
    //     // add new ones if needed
    //     float2 fullResUV = normalizedScreenPos + g_ASSAOConsts.PerPassFullResUVOffset.xy;
    //     float importance = g_ImportanceMap.SampleLevel( g_LinearClampSampler, fullResUV, 0.0 ).x;

    //     // this is to normalize SSAO_DETAIL_AO_AMOUNT across all pixel regardless of importance
    //     obscuranceSum *= (SSAO_ADAPTIVE_TAP_BASE_COUNT / (float)SSAO_MAX_TAPS) + (importance * SSAO_ADAPTIVE_TAP_FLEXIBLE_COUNT / (float)SSAO_MAX_TAPS);

    //     // load existing base values
    //     float2 baseValues = g_FinalSSAO.Load( int4( SVPosui, g_ASSAOConsts.PassIndex, 0 ) ).xy;
    //     weightSum += baseValues.y * (float)(SSAO_ADAPTIVE_TAP_BASE_COUNT * 4.0);
    //     obscuranceSum += (baseValues.x) * weightSum;

    //     // increase importance around edges
    //     float edgeCount = dot( 1.0-edgesLRTB, float4( 1.0, 1.0, 1.0, 1.0 ) );
    //     //importance += edgeCount / (float)SSAO_ADAPTIVE_TAP_FLEXIBLE_COUNT;

    //     float avgTotalImportance = (float)g_LoadCounter[0] * g_ASSAOConsts.LoadCounterAvgDiv;

    //     float importanceLimiter = saturate( g_ASSAOConsts.AdaptiveSampleCountLimit / avgTotalImportance );
    //     importance *= importanceLimiter;

    //     float additionalSampleCountFlt = SSAO_ADAPTIVE_TAP_FLEXIBLE_COUNT * importance;

    //     const float blendRange = 3.0; // use 1 to just blend the last one; use larger number to blend over more for a more smooth transition
    //     const float blendRangeInv = 1.0 / blendRange;

    //     additionalSampleCountFlt += 0.5;
    //     uint additionalSamples   = uint( additionalSampleCountFlt );
    //     uint additionalSamplesTo = min( SSAO_MAX_TAPS, additionalSamples + SSAO_ADAPTIVE_TAP_BASE_COUNT );

    //     // additional manual unroll doesn't help unfortunately
    //     [loop]
    //     for( uint i = SSAO_ADAPTIVE_TAP_BASE_COUNT; i < additionalSamplesTo; i++ )
    //     {
    //         additionalSampleCountFlt -= 1.0f;
    //         float weightMod = saturate(additionalSampleCountFlt * blendRangeInv); // slowly blend in the last few samples
    //         SSAOTap( qualityLevel, obscuranceSum, weightSum, i, rotScale, pixCenterPos, negViewspaceDir, pixelNormal, normalizedScreenPos, mipOffset, falloffCalcMulSq, weightMod, normXY, normXYLength );
    //     }
    // }

    // // early out for adaptive base - just output weight (used for the next pass)
    // if( adaptiveBase )
    // {
    //     float obscurance = obscuranceSum / weightSum;

    //     outShadowTerm   = obscurance;
    //     outEdges        = 0;
    //     outWeight       = weightSum;
    //     return;
    // }

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

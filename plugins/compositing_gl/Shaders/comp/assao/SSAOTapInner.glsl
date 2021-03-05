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

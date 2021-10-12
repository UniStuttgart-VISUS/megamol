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

void SSAOTapInner( const int qualityLevel, inout float obscuranceSum, inout float weightSum, const vec2 samplingUV, const float mipLevel, const vec3 pixCenterPos, const vec3 negViewspaceDir, vec3 pixelNormal, const float falloffCalcMulSq, const float weightMod, const int dbgTapIndex )
{
    // get depth at sample
    float viewspaceSampleZ = textureLod( g_ViewSpaceDepthSourceDepthTapSampler, samplingUV.xy, mipLevel ).x; // * g_ASSAOConsts.MaxViewspaceDepth;

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

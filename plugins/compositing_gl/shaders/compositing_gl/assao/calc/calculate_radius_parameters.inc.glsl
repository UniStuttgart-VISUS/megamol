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
    falloffCalcMulSq = -1.0f / (effectRadius*effectRadius);
}

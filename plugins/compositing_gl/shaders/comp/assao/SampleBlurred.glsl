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

vec2 SampleBlurred( vec4 inPos, vec2 coord )
{
    float packedEdges = texelFetch(g_BlurInput, ivec2( inPos.xy), 0 ).y;
    vec4 edgesLRTB    = UnpackEdges( packedEdges );

    float ssaoValue;
    float ssaoValueL;
    float ssaoValueT;
    float ssaoValueR;
    float ssaoValueB;

    // automatically done in our shader
    vec4 valuesBL = textureGather( g_BlurInput, coord - g_ASSAOConsts.HalfViewportPixelSize * 0.5 );
    vec4 valuesUR = textureGather( g_BlurInput, coord + g_ASSAOConsts.HalfViewportPixelSize * 0.5 );

    // fetch all ssaoValues around current pixel
    ssaoValue     = valuesBL.y;   // center   e.g. (5,5)                                          vUR.x
    ssaoValueL    = valuesBL.x;   // left     --> (4,5)                                   vBL.x   vBL.y   vUR.z
    ssaoValueT    = valuesUR.x;   // top      --> (5,6)                                           vBL.z
    ssaoValueR    = valuesUR.z;   // right    valuesBR.z == (6,6) --> .z = (6,5)
    ssaoValueB    = valuesBL.z;   // bottom   --> (5,4)

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

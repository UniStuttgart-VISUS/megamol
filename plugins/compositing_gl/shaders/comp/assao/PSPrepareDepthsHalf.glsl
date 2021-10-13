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

layout(local_size_x = 8, local_size_y = 8) in;

//void PSPrepareDepthsHalf( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1 )
void main()
{
    vec3 inPos = gl_GlobalInvocationID;

    ivec2 baseCoord = ivec2(inPos.xy * 2);
    float a = texelFetchOffset(g_DepthSource, baseCoord, 0, ivec2( 0, 0 ) ).x;
    float d = texelFetchOffset(g_DepthSource, baseCoord, 0, ivec2( 1, 1 ) ).x;

    float out0 = ScreenSpaceToViewSpaceDepth( a );
    float out3 = ScreenSpaceToViewSpaceDepth( d );

    imageStore(g_HalfDepths0, ivec2(inPos.xy), vec4(out0, 0.f, 0.f, 0.f));
    imageStore(g_HalfDepths3, ivec2(inPos.xy), vec4(out3, 0.f, 0.f, 0.f));
}

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

//void PSPrepareDepths( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
void main()
{
    vec3 inPos = gl_GlobalInvocationID;

#if 0   // gather can be a bit faster but doesn't work with input depth buffers that don't match the working viewport
    vec2 gatherUV = inPos.xy * g_ASSAOConsts.Viewport2xPixelSize;
    vec4 depths = textureGather(g_DepthSource, gatherUV );
    float a = depths.w;  // texelFetchOffset(g_DepthSource, ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 0, 0 ) ).x;
    float b = depths.z;  // texelFetchOffset(g_DepthSource, ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 1, 0 ) ).x;
    float c = depths.x;  // texelFetchOffset(g_DepthSource, ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 0, 1 ) ).x;
    float d = depths.y;  // texelFetchOffset(g_DepthSource, ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 1, 1 ) ).x;
#else
    ivec2 baseCoord = ivec2(inPos.xy * 2);
    float a = texelFetchOffset(g_DepthSource, baseCoord, 0, ivec2( 0, 0 ) ).x;
    float b = texelFetchOffset(g_DepthSource, baseCoord, 0, ivec2( 1, 0 ) ).x;
    float c = texelFetchOffset(g_DepthSource, baseCoord, 0, ivec2( 0, 1 ) ).x;
    float d = texelFetchOffset(g_DepthSource, baseCoord, 0, ivec2( 1, 1 ) ).x;
#endif

    float out0 = ScreenSpaceToViewSpaceDepth( a );
    float out1 = ScreenSpaceToViewSpaceDepth( b );
    float out2 = ScreenSpaceToViewSpaceDepth( c );
    float out3 = ScreenSpaceToViewSpaceDepth( d );

    ivec2 storePos = ivec2(inPos.xy);
    imageStore(g_HalfDepths0, storePos, vec4(out0, 0.f, 0.f, 0.f));
    imageStore(g_HalfDepths1, storePos, vec4(out1, 0.f, 0.f, 0.f));
    imageStore(g_HalfDepths2, storePos, vec4(out2, 0.f, 0.f, 0.f));
    imageStore(g_HalfDepths3, storePos, vec4(out3, 0.f, 0.f, 0.f));
}

// AMD Cauldron code
//
// Copyright(c) 2020 Advanced Micro Devices, Inc.All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

layout(local_size_x = 8, local_size_y = 8) in;

uniform int mip_level;
uniform sampler2D g_input_tx2D;
layout(rgba8, binding = 0) uniform writeonly image2D g_output_tx2D;

//--------------------------------------------------------------------------------------
// Texture definitions
//--------------------------------------------------------------------------------------
// Texture2D        inputTex         :register(t0);
// SamplerState     samLinearMirror  :register(s0);

//--------------------------------------------------------------------------------------
// Main function
//--------------------------------------------------------------------------------------

ivec2 offsets[9] = {
    ivec2( 1, 1), ivec2( 0, 1), ivec2(-1, 1),
    ivec2( 1, 0), ivec2( 0, 0), ivec2(-1, 0),
    ivec2( 1,-1), ivec2( 0,-1), ivec2(-1,-1)
    };

//float4 mainPS(VERTEX Input) : SV_Target
void main()
{
    ivec2 in_pos = ivec2(gl_GlobalInvocationID.xy);

    // gaussian like downsampling

    vec4 color = vec4(0.0);

    if (mip_level==0)
    {
        for(int i=0;i<9;i++) {
            //color += log(max(inputTex.Sample(samLinearMirror, Input.vTexcoord + (2 * u_invSize * offsets[i])), float4(0.01, 0.01, 0.01, 0.01) ));
            color += log(max(0.01 + texelFetch(g_input_tx2D, 2 * in_pos + offsets[i], 0), vec4(0.01)));
        }
        color = exp(color / 9.0f);
        imageStore(g_output_tx2D, in_pos, color);
    }
    else
    {
        for(int i=0;i<9;i++) {
            //color += inputTex.Sample(samLinearMirror, Input.vTexcoord + (2 * u_invSize * offsets[i]) );
            color += texelFetch(g_input_tx2D, 2 * in_pos + offsets[i], mip_level);
        }
        color = color / 9.0f;
        imageStore(g_output_tx2D, in_pos, color);
    }
}

#version 450
#extension GL_ARB_shading_language_420pack : enable
#extension GL_ARB_shading_language_include : enable
// FidelityFX Super Resolution Sample
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

layout(local_size_x = 8, local_size_y = 8) in;

//-----------------------------------------------------------------------------
// UNIFORMS
layout(std430, binding = 0) readonly buffer easu_const_buffer
{
	uvec4 Const0;
	uvec4 Const1;
	uvec4 Const2;
	uvec4 Const3;
	uvec4 Sample;
};

#define A_GPU 1
#define A_GLSL 1
#define SAMPLE_EASU 1
#define SAMPLE_RCAS 0
#define SAMPLE_BILINEAR 0

uniform sampler2D InputTexture;
layout(rgba8, binding = 1) uniform writeonly image2D OutputTexture;

#include "3rd/ffx_a.h"
#if SAMPLE_EASU
    #define FSR_EASU_F 1
    AF4 FsrEasuRF(AF2 p) { AF4 res = textureGather(InputTexture, p, 0); return res; }
    AF4 FsrEasuGF(AF2 p) { AF4 res = textureGather(InputTexture, p, 1); return res; }
    AF4 FsrEasuBF(AF2 p) { AF4 res = textureGather(InputTexture, p, 2); return res; }
#endif
#if SAMPLE_RCAS
    #define FSR_RCAS_F
    AF4 FsrRcasLoadF(ASU2 p) { return texelFetch(InputTexture, ASU2(p), 0); }
    void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}
#endif

#include "3rd/ffx_fsr1.h"

void CurrFilter(AU2 pos)
{
#if SAMPLE_BILINEAR
	AF2 pp = (AF2(pos) * AF2_AU2(Const0.xy) + AF2_AU2(Const0.zw)) * AF2_AU2(Const1.xy) + AF2(0.5, -0.5) * AF2_AU2(Const1.zw);
	imageStore(OutputTexture, ASU2(pos), textureLod(InputTexture, pp, 0.0));
#endif
#if SAMPLE_EASU
    AF3 c;
    FsrEasuF(c, pos, Const0, Const1, Const2, Const3);
    if( Sample.x == 1 )
        c *= c;
    imageStore(OutputTexture, ASU2(pos), AF4(c, 1));
#endif
#if SAMPLE_RCAS
    AF3 c;
    FsrRcasF(c.r, c.g, c.b, pos, Const0);
    if( Sample.x == 1 )
        c *= c;
    imageStore(OutputTexture, ASU2(pos), AF4(c, 1));
#endif
}

void main() {
    vec4 red = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 green = vec4(0.0, 1.0, 0.0, 1.0);
    vec4 blue = vec4(0.0, 0.0, 1.0, 1.0);
    vec4 yellow = vec4(1.0, 1.0, 0.0, 1.0);

    uvec2 in_pos = uvec2(gl_GlobalInvocationID.xy);

    CurrFilter(2 * in_pos);
	CurrFilter(2 * in_pos + uvec2(1, 0));
	CurrFilter(2 * in_pos + uvec2(0, 1));
	CurrFilter(2 * in_pos + uvec2(1, 1));

    //imageStore(OutputTexture, 2 * in_pos + ivec2(0, 0), red);
    //imageStore(OutputTexture, 2 * in_pos + ivec2(1, 0), green);
    //imageStore(OutputTexture, 2 * in_pos + ivec2(0, 1), blue);
    //imageStore(OutputTexture, 2 * in_pos + ivec2(1, 1), yellow);

    // // Do remapping of local xy in workgroup for a more PS-like swizzle pattern.
	// AU2 gxy = ARmp8x8(gl_LocalInvocationID.x) + AU2(gl_WorkGroupID.x << 4u, gl_WorkGroupID.y << 4u);
	// CurrFilter(gxy);
    // imageStore(OutputTexture, ivec2(gxy), red);
	// gxy.x += 8u;
	// CurrFilter(gxy);
    // imageStore(OutputTexture, ivec2(gxy), green);
	// gxy.y += 8u;
	// CurrFilter(gxy);
    // imageStore(OutputTexture, ivec2(gxy), blue);
	// gxy.x -= 8u;
	// CurrFilter(gxy);
    // imageStore(OutputTexture, ivec2(gxy), yellow);



}

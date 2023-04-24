/**
 * Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
 * Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
 * Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
 * Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
 * Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to
 * do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software. As clarification, there
 * is no requirement that the copyright notice and permission be included in
 * binary distributions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

#include "compositing_gl/AntiAliasing/preset_uniforms.inc.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

//-----------------------------------------------------------------------------
// UNIFORMS
uniform int SMAA_REPROJECTION;
uniform sampler2D g_colorTex;
uniform sampler2D g_blendingWeightsTex;
uniform sampler2D g_velocityTex;
#if defined OUT32F
layout(rgba32f) writeonly uniform image2D g_output;
#endif
#if defined OUT16HF
layout(rgba16f) writeonly uniform image2D g_output;
#endif
#if defined OUT8NB
layout(rgba8_snorm) writeonly uniform image2D g_output;
#endif


/**
 * Conditional move:
 */
void SMAAMovc(bvec2 cond, inout vec2 variable, vec2 value) {
    if (cond.x) variable.x = value.x;
    if (cond.y) variable.y = value.y;
}

void SMAAMovc(bvec4 cond, inout vec4 variable, vec4 value) {
    SMAAMovc(cond.xy, variable.xy, value.xy);
    SMAAMovc(cond.zw, variable.zw, value.zw);
}

//-----------------------------------------------------------------------------
// Neighborhood Blending Pixel Shader (Third Pass)
vec4 SMAANeighborhoodBlendingPS(vec2 texcoord,
                                  vec4 offset,
                                  sampler2D colorTex,
                                  sampler2D blendTex,
                                  sampler2D velocityTex
                                  ) {
    // Fetch the blending weights for current pixel:
    vec4 a;
    a.x = texture(blendTex, offset.xy).a; // Right
    a.y = texture(blendTex, offset.zw).g; // Top
    a.wz = texture(blendTex, texcoord).xz; // Bottom / Left

    // Is there any blending weight with a value greater than 0.0?
    if (dot(a, vec4(1.0, 1.0, 1.0, 1.0)) < 1e-5) {
        vec4 color = textureLod(colorTex, texcoord, 0.0);

        if( SMAA_REPROJECTION == 1 ) {
            vec2 velocity = textureLod(velocityTex, texcoord, 0.0).rg;

            // Pack velocity into the alpha channel:
            color.a = sqrt(5.0 * length(velocity));
        }

        return color;
    } else {
        bool h = max(a.x, a.z) > max(a.y, a.w); // max(horizontal) > max(vertical)

        // Calculate the blending offsets:
        vec4 blendingOffset = vec4(0.0, a.y, 0.0, a.w);
        vec2 blendingWeight = a.yw;
        SMAAMovc(bvec4(h, h, h, h), blendingOffset, vec4(a.x, 0.0, a.z, 0.0));
        SMAAMovc(bvec2(h, h), blendingWeight, a.xz);
        blendingWeight /= dot(blendingWeight, vec2(1.0, 1.0));

        // Calculate the texture coordinates:
        vec4 blendingCoord = fma(blendingOffset, vec4(g_SMAAConsts.SMAA_RT_METRICS.xy, -g_SMAAConsts.SMAA_RT_METRICS.xy), texcoord.xyxy);

        // We exploit bilinear filtering to mix current pixel with the chosen
        // neighbor:
        vec4 color = blendingWeight.x * textureLod(colorTex, blendingCoord.xy, 0.0);
        color += blendingWeight.y * textureLod(colorTex, blendingCoord.zw, 0.0);

        if( SMAA_REPROJECTION == 1 ) {
            // Antialias velocity for proper reprojection in a later stage:
            vec2 velocity = blendingWeight.x * textureLod(velocityTex, blendingCoord.xy, 0.0).rg;
            velocity += blendingWeight.y * textureLod(velocityTex, blendingCoord.zw, 0.0).rg;

            // Pack velocity into the alpha channel:
            color.a = sqrt(5.0 * length(velocity));
        }

        return color;
    }
}

void main() {
    vec3 inPos = gl_GlobalInvocationID.xyz;

    // Minor: could be optimized with rt_metrics I believe, see shaders from assao
    vec2 texCoords = (2.f * inPos.xy + vec2(1.f)) / (2.f * vec2(g_SMAAConsts.SMAA_RT_METRICS.zw));

    vec4 offset = fma(g_SMAAConsts.SMAA_RT_METRICS.xyxy, vec4( 1.0, 0.0, 0.0, 1.0), texCoords.xyxy);

    vec4 final = SMAANeighborhoodBlendingPS(texCoords,
        offset,
        g_colorTex,
        g_blendingWeightsTex,
        g_velocityTex
    );

    imageStore(g_output, ivec2(inPos.xy), final);
}

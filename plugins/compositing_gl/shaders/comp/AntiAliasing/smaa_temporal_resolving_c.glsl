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

layout(local_size_x = 8, local_size_y = 8) in;

//-----------------------------------------------------------------------------
// UNIFORMS
uniform int SMAA_REPROJECTION;
uniform sampler2D g_currColorTex;
uniform sampler2D g_prevColorTex;
uniform sampler2D g_velocityTex;

layout(rgba16f, binding = 0) uniform writeonly image2D g_outputTex;


void main() {

    vec3 inPos = gl_GlobalInvocationID.xyz;

    vec2 texCoords = (2.f * inPos.xy + vec2(1.f)) / (2.f * vec2(g_SMAAConsts.SMAA_RT_METRICS.zw));

    if (SMAA_REPROJECTION == 1) {
        // Velocity is assumed to be calculated for motion blur, so we need to
        // inverse it for reprojection:
        vec2 velocity = -texture(g_velocityTex, texCoords).rg;

        // Fetch current pixel:
        vec4 current = texture(g_currColorTex, texCoords);

        // Reproject current coordinates and fetch previous pixel:
        vec4 previous = texture(g_prevColorTex, texCoords + velocity);

        // Attenuate the previous pixel if the velocity is different:
        float delta = abs(current.a * current.a - previous.a * previous.a) / 5.0;
        float weight = 0.5 * clamp(1.0 - sqrt(delta) * g_SMAAConsts.SMAA_REPROJECTION_WEIGHT_SCALE, 0.0, 1.0);

        // Blend the pixels according to the calculated weight:
        vec4 result = mix(current, previous, weight);
        imageStore(g_outputTex, ivec2(inPos.xy), result);
    }
    else {
        // Just blend the pixels:
        vec4 current = texture(g_currColorTex, texCoords);
        vec4 previous = texture(g_prevColorTex, texCoords);
        vec4 result = mix(current, previous, 0.5);
        imageStore(g_outputTex, ivec2(inPos.xy), result);
    }
}

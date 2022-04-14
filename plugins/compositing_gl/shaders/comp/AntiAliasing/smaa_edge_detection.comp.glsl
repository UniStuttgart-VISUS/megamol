#version 450

#include "comp/AntiAliasing/preset_uniforms.inc.glsl"

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
uniform int technique;
uniform sampler2D g_depthTex;
uniform sampler2D g_colorTex;
layout(rgba8, binding = 0) uniform image2D g_edgesTex;


//-----------------------------------------------------------------------------
// Misc functions
/**
 * Gathers current pixel, and the top-left neighbors.
 */
vec3 SMAAGatherNeighbours(vec2 texcoord,
                            vec4 offset[3],
                            sampler2D tex) {
    return textureGather(tex, texcoord + g_SMAAConsts.SMAA_RT_METRICS.xy * vec2(-0.5, -0.5)).grb;
}

/**
 * Adjusts the threshold by means of predication.
 */
vec2 SMAACalculatePredicatedThreshold(vec2 texcoord,
                                        vec4 offset[3],
                                        sampler2D predicationTex) {
    vec3 neighbours = SMAAGatherNeighbours(texcoord, offset, predicationTex);
    vec2 delta = abs(neighbours.xx - neighbours.yz);
    vec2 edges = step(g_SMAAConsts.SMAA_PREDICATION_THRESHOLD, delta);
    return g_SMAAConsts.SMAA_PREDICATION_SCALE * g_SMAAConsts.SMAA_THRESHOLD * (1.0 - g_SMAAConsts.SMAA_PREDICATION_STRENGTH * edges);
}
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Edge Detection Pixel Shaders (First Pass)

/**
 * Luma Edge Detection
 *
 * IMPORTANT NOTICE: luma edge detection requires gamma-corrected colors, and
 * thus 'colorTex' should be a non-sRGB texture.
 */
vec2 SMAALumaEdgeDetectionPS(vec2 texcoord,
                               vec4 offset[3],
                               sampler2D colorTex
                               ) {
    // Calculate the threshold:
    vec2 threshold = vec2(g_SMAAConsts.SMAA_THRESHOLD);

    // Calculate lumas:
    vec3 weights = vec3(0.2126, 0.7152, 0.0722);
    float L = dot(texture(colorTex, texcoord).rgb, weights);

    float Lleft = dot(texture(colorTex, offset[0].xy).rgb, weights);
    float Ltop  = dot(texture(colorTex, offset[0].zw).rgb, weights);

    // We do the usual threshold:
    vec4 delta;
    delta.xy = abs(L - vec2(Lleft, Ltop));
    vec2 edges = step(threshold, delta.xy);

    // Then discard if there is no edge:
    if (dot(edges, vec2(1.0, 1.0)) == 0.0)
        return vec2(0.0);

    // Calculate right and bottom deltas:
    float Lright = dot(texture(colorTex, offset[1].xy).rgb, weights);
    float Lbottom  = dot(texture(colorTex, offset[1].zw).rgb, weights);
    delta.zw = abs(L - vec2(Lright, Lbottom));

    // Calculate the maximum delta in the direct neighborhood:
    vec2 maxDelta = max(delta.xy, delta.zw);

    // Calculate left-left and top-top deltas:
    float Lleftleft = dot(texture(colorTex, offset[2].xy).rgb, weights);
    float Ltoptop = dot(texture(colorTex, offset[2].zw).rgb, weights);
    delta.zw = abs(vec2(Lleft, Ltop) - vec2(Lleftleft, Ltoptop));

    // Calculate the final maximum delta:
    maxDelta = max(maxDelta.xy, delta.zw);
    float finalDelta = max(maxDelta.x, maxDelta.y);

    // Local contrast adaptation:
    edges.xy *= step(finalDelta, g_SMAAConsts.SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta.xy);

    return edges;
}

/**
 * Color Edge Detection
 *
 * IMPORTANT NOTICE: color edge detection requires gamma-corrected colors, and
 * thus 'colorTex' should be a non-sRGB texture.
 */
vec2 SMAAColorEdgeDetectionPS(vec2 texcoord,
                                vec4 offset[3],
                                sampler2D colorTex
                                ) {
    // Calculate the threshold:
    vec2 threshold = vec2(g_SMAAConsts.SMAA_THRESHOLD);

    // Calculate color deltas:
    vec4 delta;
    vec3 C = texture(colorTex, texcoord).rgb;

    vec3 Cleft = texture(colorTex, offset[0].xy).rgb;
    vec3 t = abs(C - Cleft);
    delta.x = max(max(t.r, t.g), t.b);

    vec3 Ctop  = texture(colorTex, offset[0].zw).rgb;
    t = abs(C - Ctop);
    delta.y = max(max(t.r, t.g), t.b);

    // We do the usual threshold:
    vec2 edges = step(threshold, delta.xy);

    // Then discard if there is no edge:
    if (dot(edges, vec2(1.0, 1.0)) == 0.0)
        return vec2(0.0);

    // Calculate right and bottom deltas:
    vec3 Cright = texture(colorTex, offset[1].xy).rgb;
    t = abs(C - Cright);
    delta.z = max(max(t.r, t.g), t.b);

    vec3 Cbottom  = texture(colorTex, offset[1].zw).rgb;
    t = abs(C - Cbottom);
    delta.w = max(max(t.r, t.g), t.b);

    // Calculate the maximum delta in the direct neighborhood:
    vec2 maxDelta = max(delta.xy, delta.zw);

    // Calculate left-left and top-top deltas:
    vec3 Cleftleft  = texture(colorTex, offset[2].xy).rgb;
    t = abs(C - Cleftleft);
    delta.z = max(max(t.r, t.g), t.b);

    vec3 Ctoptop = texture(colorTex, offset[2].zw).rgb;
    t = abs(C - Ctoptop);
    delta.w = max(max(t.r, t.g), t.b);

    // Calculate the final maximum delta:
    maxDelta = max(maxDelta.xy, delta.zw);
    float finalDelta = max(maxDelta.x, maxDelta.y);

    // Local contrast adaptation:
    edges.xy *= step(finalDelta, g_SMAAConsts.SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta.xy);

    return edges;
}

/**
 * Depth Edge Detection
 */
vec2 SMAADepthEdgeDetectionPS(vec2 texcoord,
                                vec4 offset[3],
                                sampler2D depthTex) {
    vec3 neighbours = SMAAGatherNeighbours(texcoord, offset, depthTex);
    vec2 delta = abs(neighbours.xx - vec2(neighbours.y, neighbours.z));
    vec2 edges = step(g_SMAAConsts.SMAA_DEPTH_THRESHOLD, delta);

    if (dot(edges, vec2(1.0, 1.0)) == 0.0) {
        //discard;
        edges = vec2(0.0);
    }

    return edges;
}


void main() {
    vec3 inPos = gl_GlobalInvocationID.xyz;

    // Minor: could be optimized I believe
    vec2 texCoords = (2.f * inPos.xy + vec2(1.f)) / (2.f * vec2(g_SMAAConsts.SMAA_RT_METRICS.zw));

    vec4 offset[3];
    offset[0] = fma(g_SMAAConsts.SMAA_RT_METRICS.xyxy, vec4(-1.0, 0.0, 0.0, -1.0), texCoords.xyxy);
    offset[1] = fma(g_SMAAConsts.SMAA_RT_METRICS.xyxy, vec4( 1.0, 0.0, 0.0,  1.0), texCoords.xyxy);
    offset[2] = fma(g_SMAAConsts.SMAA_RT_METRICS.xyxy, vec4(-2.0, 0.0, 0.0, -2.0), texCoords.xyxy);

    vec2 edges = vec2(0.0);

    // luma
    if(technique == 0) {
        edges = SMAALumaEdgeDetectionPS(texCoords, offset, g_colorTex);
    }
    // color
    else if(technique == 1) {
        edges = SMAAColorEdgeDetectionPS(texCoords, offset, g_colorTex);
    }
    // depth
    else if(technique == 2) {
        edges = SMAADepthEdgeDetectionPS(texCoords, offset, g_depthTex);
    }

    imageStore(g_edgesTex, ivec2(inPos.xy), vec4(edges, rt_snd.x, 0.0));
}

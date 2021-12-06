/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

layout(local_size_x = 8, local_size_y = 8) in;

uniform mat4 prevProjMx;
uniform mat4 prevViewMx;
uniform mat4 currProjMx;
uniform mat4 currViewMx;

uniform vec2 jitter;

uniform sampler2D g_currDepthtex;
uniform sampler2D g_prevDepthtex;

layout(rg16f, binding = 0) uniform writeonly image2D velocityTx2D;

void main() {
    vec2 inPos = gl_GlobalInvocationID.xy;

    vec2 viewport = imageSize(velocityTx2D);

    // use texturefetch
    float currDepth = texelFetch(g_currDepthtex, ivec2(inPos), 0).r;
    float prevDepth = texelFetch(g_prevDepthtex, ivec2(inPos), 0).r;

    vec4 currPos = vec4(inPos / viewport, currDepth, 1.0);
    vec4 prevPos = vec4(inPos / viewport, prevDepth, 1.0);

    // ndc
    currPos.xy = currPos.xy * 2.0 - vec2(1.0);
    prevPos.xy = prevPos.xy * 2.0 - vec2(1.0);

    // world space
    currPos = inverse(currProjMx) * currPos;
    prevPos = inverse(prevProjMx) * prevPos;

    currPos = currPos / currPos.w;
    prevPos = prevPos / prevPos.w;

    currPos = inverse(currViewMx) * currPos;
    prevPos = inverse(prevViewMx) * prevPos;

    // translate according to jitter
    currPos.xy = currPos.xy + vec2(2.0 * jitter.x / viewport.x, 2.0 * jitter.y / viewport.y);
    prevPos.xy = prevPos.xy + vec2(2.0 * jitter.x / viewport.x, 2.0 * jitter.y / viewport.y);

    // re-project
    currPos = currProjMx * currViewMx * vec4(currPos.xyz, 1.0);
    prevPos = prevProjMx * prevViewMx * vec4(prevPos.xyz, 1.0);

    // perspective division
    currPos = currPos / currPos.w;
    prevPos = prevPos / prevPos.w;

    // Note from original code https://github.com/iryoku/smaa :
    // Positions in projection space are in [-1, 1] range, while texture
    // coordinates are in [0, 1] range. So, we divide by 2 to get velocities in
    // the scale:
    currPos.xy = currPos.xy * vec2(0.5);
    prevPos.xy = prevPos.xy * vec2(0.5);

    // calc velocity
    vec2 velocity = (currPos.xy - prevPos.xy);

    imageStore(velocityTx2D, ivec2(inPos), vec4(velocity, 0.0, 0.0));
}

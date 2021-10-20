/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

uniform mat4 prevViewProjMx;
uniform mat4 currViewProjMx;

uniform vec2 jitter;

uniform sampler2D g_currDepthtex;
uniform sampler2D g_prevDepthtex;

layout(rg8, binding = 0) uniform writeonly image2D velocityTx2D;

void main() {
    vec2 inPos = gl_GlobalInvocationID.xy;

    vec2 viewport = imageSize(velocityTx2D);

    vec2 texCoords = (2.f * inPos + vec2(1.f)) / (2.f * vec2(viewport));

    float currDepth = texture(g_currDepthtex, texCoords);
    float prevDepth = texture(g_prevDepthtex, texCoords);

    vec4 currPos = vec3(inPos, currDepth, 1.0);
    vec4 prevPos = vec3(inPos, prevDepth, 1.0);

    // ndc
    currPos.xy = currPos.xy * 2.0 - vec2(1.0);
    prevPos.xy = prevPos.xy * 2.0 - vec2(1.0);

    //currPos = currPos / currW; ??
    //prevPos = prevPos / prevW; ??

    // world space
    currPos = inv(currViewProjMx) * currPos;
    prevPos = inv(prevViewProjMx) * prevPos;

    // translate according to jitter
    currPos.xy = currPos.xy + vec2(2.0 * jitter.x / viewport.x, 2.0 * jitter.y / viewport.y);
    prevPos.xy = prevPos.xy + vec2(2.0 * jitter.x / viewport.x, 2.0 * jitter.y / viewport.y);

    // re-project
    currPos = currViewProjMx * vec4(currPos.xyz, 1.0);
    prevPos = prevViewProjMx * vec4(prevPos.xyz, 1.0);

    // perspective division
    currPos = currPos / currPos.w;
    prevPos = prevPos / prevPos.w;

    // Note from original code https://github.com/iryoku/smaa:
    // Positions in projection space are in [-1, 1] range, while texture
    // coordinates are in [0, 1] range. So, we divide by 2 to get velocities in
    // the scale:
    currPos.xy = currPos.xy * vec2(0.5);
    prevPos.xy = prevPos.xy * vec2(0.5);

    // calc velocity
    vec2 velocity = (currPos.xy - prevPos.xy);

    imageStore(velocityTx2D, inPos, vec4(velocity, 0.0, 0.0));
}
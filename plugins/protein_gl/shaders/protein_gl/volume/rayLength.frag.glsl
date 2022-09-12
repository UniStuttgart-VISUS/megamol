#version 120

// Copyright (c) 2009  Martin Falk <falk@vis.uni-stuttgart.de>
//                     Visualization Research Center (VISUS),
//                     Universitaet Stuttgart, Germany
//                     http://www.vis.uni-stuttgart.de/~falkmn/
//      modified 2010  Michael Krone <kroneml@vis.uni-stuttgart.de>
// This program may be distributed, modified, and used free of charge
// as long as this copyright notice is included in its original form.
// Commercial use is strictly prohibited.

#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D sourceTex;
uniform sampler2D depthTex;
uniform vec2 screenResInv;
uniform vec2 zNearFar;

uniform float scale;

varying vec3 objPos;

float reconstructDepth(float z) {
    //float Zn = zNearFar.x/scale;
    //float Zf = zNearFar.y/scale;
    float Zn = zNearFar.x;
    float Zf = zNearFar.y;

    return Zn*Zf / (Zf - z*(Zf - Zn));
}

void main(void) {
    vec4 rayStart = texelFetch2D( sourceTex, ivec2( gl_FragCoord.xy), 0 );

    vec3 rayDir = objPos - rayStart.xyz;
    float rayLen = length(rayDir);

    rayDir = normalize(rayDir);

    float sceneDepthDev = texelFetch2D( depthTex, ivec2( gl_FragCoord.xy), 0 ).x;
    float sceneDepth = reconstructDepth( sceneDepthDev);
    // correction factor for depth projection  dot(rayDir, viewDir)
    float projection = abs(dot(rayDir, normalize(gl_ModelViewMatrixInverse[2].xyz)));
    // compute distance between rayStart and camera
    float rayStartLen = length(rayStart.xyz - gl_ModelViewMatrixInverse[3].xyz);
    // depth difference between volume begin and geometry
    float x = (sceneDepth)/projection - rayStartLen;

    rayLen = min(x, rayLen);
    rayLen *= rayStart.w;

    gl_FragData[0] = vec4( rayDir, rayLen);
    gl_FragData[1] = vec4( rayStartLen);
}

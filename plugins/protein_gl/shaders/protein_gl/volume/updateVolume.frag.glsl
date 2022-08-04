#version 120

// Copyright (c) 2009  Martin Falk <falk@vis.uni-stuttgart.de>
//                     Visualization Research Center (VISUS),
//                     Universitaet Stuttgart, Germany
//                     http://www.vis.uni-stuttgart.de/~falkmn/
//      modified 2010  Michael Krone <kroneml@vis.uni-stuttgart.de>
// This program may be distributed, modified, and used free of charge
// as long as this copyright notice is included in its original form.
// Commercial use is strictly prohibited.

uniform float filterRadius;
uniform float sliceDepth;
uniform vec3 invVolRes;
uniform vec3 scaleVolInv;

uniform float densityScale;

varying vec3 splatCenter;
varying float scale;
varying vec3 color;

float computeDensity( vec3 pos) {
    float d = distance( pos, splatCenter.xyz);
    //float density = ( 1.0 - smoothstep( 0.0, 1.0*filterRadius*scale, d)) * densityScale;
    // gaussian function
    float density = filterRadius * exp( -(d * d)/(2 * scale * scale));
    //if( d < scale ) density = 1.0; else density = 0.0; // DEBUG
    return density;
}

float computeColorDensity( vec3 pos) {
    float d = distance( pos, splatCenter.xyz);
    //float density = ( 1.0 - smoothstep( 0.2*filterRadius*scale, 1.0*filterRadius*scale, d)) * densityScale;
    // gaussian function
    float density = filterRadius * exp( -(d * d)/(2 * scale * scale)) * densityScale;
    return density;
}

void main(void) {
    vec3 pos = vec3( gl_FragCoord.xy * invVolRes.xy, sliceDepth) * scaleVolInv;

    float density = computeDensity( pos);
    //gl_FragColor = vec4( density);

    // DEBUG ...
    //gl_FragColor = vec4( color * density * scale, density);
    gl_FragColor = vec4( color * computeColorDensity( pos) * scale, density);
    // ... DEBUG
}

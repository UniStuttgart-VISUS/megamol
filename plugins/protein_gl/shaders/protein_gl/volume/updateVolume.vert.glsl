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
uniform vec3 scaleVol;
uniform vec3 scaleVolInv;
uniform float sliceDepth;
uniform vec3 translate;
uniform float volSize = 128.0;

varying float scale;
varying vec3 splatCenter;
varying vec3 color;

void main(void) {
    color = gl_Color.rgb;

    splatCenter = gl_Vertex.xyz - translate;
    vec2 devCoord = 2.0 * splatCenter.xy * scaleVol.xy - 1.0;
    //vec2 devCoord = splatCenter.xy * scaleVol.xy;

    //scale = gl_Vertex.w*30.0;
    scale = gl_Vertex.w;

    float d = length( vec3( splatCenter.xy, sliceDepth*scaleVolInv.z) - splatCenter);
    if( d > filterRadius*scale) {
        gl_Position = vec4( 0.0, 0.0, 1000.0, 1.0);
    } else {
        gl_Position = vec4( devCoord, 0.0, 1.0);
    }

    gl_PointSize = 2.0 * filterRadius * max( scaleVol.x, scaleVol.y) * volSize * scale;
}

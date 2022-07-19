#version 120

// Copyright (c) 2009  Martin Falk <falk@vis.uni-stuttgart.de>
//                     Visualization Research Center (VISUS),
//                     Universitaet Stuttgart, Germany
//                     http://www.vis.uni-stuttgart.de/~falkmn/
//      modified 2010  Michael Krone <kroneml@vis.uni-stuttgart.de>
// This program may be distributed, modified, and used free of charge
// as long as this copyright notice is included in its original form.
// Commercial use is strictly prohibited.

varying vec3 lightPos;
varying vec3 fillLightPos;

void main(void) {
    gl_FrontColor = gl_Color;
    gl_Position = gl_Vertex;

    // calculate light position
    lightPos = ( gl_ModelViewMatrixInverse * vec4( 60.0, 50.0, 100.0, 1.0)).xyz;
    fillLightPos = ( gl_ModelViewMatrixInverse * vec4(-60.0, -20.0, 50.0, 1.0)).xyz;
}

#version 120

// Copyright (c) 2009  Martin Falk <falk@vis.uni-stuttgart.de>
//                     Visualization Research Center (VISUS),
//                     Universitaet Stuttgart, Germany
//                     http://www.vis.uni-stuttgart.de/~falkmn/
//      modified 2010  Michael Krone <kroneml@vis.uni-stuttgart.de>
// This program may be distributed, modified, and used free of charge
// as long as this copyright notice is included in its original form.
// Commercial use is strictly prohibited.

varying vec3 objPos;

void main(void) {
    // transform vertices from camera space to object space
    objPos = ( gl_ModelViewMatrixInverse * gl_Vertex).xyz;
    gl_ClipVertex = gl_Vertex;
    gl_Position = gl_ProjectionMatrix * gl_Vertex;

    gl_FrontColor = vec4( objPos, gl_Color.a);
}

#version 120

// Copyright (c) 2009  Martin Falk <falk@vis.uni-stuttgart.de>
//                     Visualization Research Center (VISUS),
//                     Universitaet Stuttgart, Germany
//                     http://www.vis.uni-stuttgart.de/~falkmn/
//      modified 2010  Michael Krone <kroneml@vis.uni-stuttgart.de>
// This program may be distributed, modified, and used free of charge
// as long as this copyright notice is included in its original form.
// Commercial use is strictly prohibited.

uniform vec3 translate = vec3(0.0);
varying vec3 objPos;

void main(void) {
    objPos = gl_Vertex.xyz - translate;
    //vec4 v = gl_Vertex; // + vec4( translate, 0.0);
    vec4 v = gl_Vertex - vec4( translate, 0.0);
    gl_ClipVertex = gl_ModelViewMatrix * v;
    gl_Position = gl_ModelViewProjectionMatrix * v;

    gl_FrontColor = vec4( objPos, gl_Color.a);
}

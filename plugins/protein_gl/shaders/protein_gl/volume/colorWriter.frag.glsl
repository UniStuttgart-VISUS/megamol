#version 120

// Copyright (c) 2011  Michael Krone <kroneml@vis.uni-stuttgart.de>
//                     Visualization Research Center (VISUS),
//                     Universitaet Stuttgart, Germany
//                     http://www.vis.uni-stuttgart.de/~kroneml/
// This program may be distributed, modified, and used free of charge
// as long as this copyright notice is included in its original form.
// Commercial use is strictly prohibited.

uniform vec4 color;

void main(void) {
    gl_FragColor = color;
}

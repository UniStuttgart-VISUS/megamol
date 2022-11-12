// Copyright (c) 2011  Michael Krone <kroneml@vis.uni-stuttgart.de>
//                     Visualization Research Center (VISUS),
//                     University of Stuttgart, Germany
//                     http://www.vis.uni-stuttgart.de/~kroneml/

#version 120

varying float density;

void main(void) {
    gl_FragColor = vec4( density, density, density, 1.0);
}

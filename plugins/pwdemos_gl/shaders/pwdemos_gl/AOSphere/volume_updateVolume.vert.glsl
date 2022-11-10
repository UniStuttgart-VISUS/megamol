// Copyright (c) 2011  Michael Krone <kroneml@vis.uni-stuttgart.de>
//                     Visualization Research Center (VISUS),
//                     University of Stuttgart, Germany
//                     http://www.vis.uni-stuttgart.de/~kroneml/

#version 120

uniform vec3 minOS;
uniform vec3 rangeOS;
uniform float genFac;
uniform float voxelVol;
uniform vec3 volSize;
uniform float sliceDepth;
uniform float radius;

varying float density;

void main(void) {
    // multiple VBO version
    //int d = int( floor( sliceDepth + 0.5));

    vec3 pos = clamp( floor( ( ( gl_Vertex.xyz - minOS) / rangeOS ) * volSize), vec3( 0.0), volSize );

    float spVol;
    if( radius < 0.0 )
      spVol = ( 4.0 / 3.0) * 3.14159265 * gl_Vertex.w * gl_Vertex.w * gl_Vertex.w;
    else
      spVol = ( 4.0 / 3.0) * 3.14159265 * radius * radius * radius;

    // multiple VBO version
    /*
    if( d != int( pos.z + 1.0) ) {
        gl_Position = vec4( 0.0, 0.0, 1000.0, 0.0);
        density = 0.0;
    } else {
        gl_Position = vec4(
            2.0 * ( pos.xy + vec2( 1.0)) / ( volSize.xy + vec2( 2.0)) - vec2( 1.0),
            0.0, 1.0);
        density = (spVol / voxelVol) * genFac;
    }
    */
    // multiple VBO version
    gl_Position = vec4(
        2.0 * ( pos.xy + vec2( 1.0)) / ( volSize.xy + vec2( 2.0)) - vec2( 1.0),
        0.0, 1.0);
    density = (spVol / voxelVol) * genFac;

}

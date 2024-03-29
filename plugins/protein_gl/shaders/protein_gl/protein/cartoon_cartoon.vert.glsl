/*
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS) / Michael Krone. Alle Rechte vorbehalten.
 */

//#version 120

varying vec4 diffuse,ambient;
varying vec3 normal,lightDir,halfVector;

void main(void)
{
    gl_FrontColor=gl_Color;
    gl_BackColor=gl_Color;

    // do not ftransform(), geometry shader needs the original vertices
    gl_Position= gl_Vertex;
}

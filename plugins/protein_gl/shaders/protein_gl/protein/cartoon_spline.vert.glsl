/*
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS) / Michael Krone. Alle Rechte vorbehalten.
 */

//#version 120

void main(void)
{
    gl_FrontColor = gl_Color;
    gl_BackColor = gl_Color;
    gl_FrontSecondaryColor = gl_SecondaryColor;
    gl_BackSecondaryColor = gl_SecondaryColor;

    // do not ftransform(), geometry shader needs the original vertices
    gl_Position = gl_Vertex;
}

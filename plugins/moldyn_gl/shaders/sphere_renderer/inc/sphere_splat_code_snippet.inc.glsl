
    inColor = vec4(theBuffer[gl_VertexID + instanceOffset].r,
                       theBuffer[gl_VertexID + instanceOffset].g,
                       theBuffer[gl_VertexID + instanceOffset].b,
                       theBuffer[gl_VertexID + instanceOffset].a); 
    inPosition = vec4(theBuffer[gl_VertexID + instanceOffset].posX,
                 theBuffer[gl_VertexID + instanceOffset].posY,
                 theBuffer[gl_VertexID + instanceOffset].posZ, 1.0); 
    rad = theBuffer[gl_VertexID + instanceOffset].posR;
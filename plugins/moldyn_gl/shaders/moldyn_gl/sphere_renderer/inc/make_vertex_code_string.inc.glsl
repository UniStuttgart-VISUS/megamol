#ifdef VERTDATA_NONE
#endif // VERTDATA_NONE


#ifdef VERTDATA_FLOAT_XYZ

#ifdef INTERLEAVED
    inPosition = vec4(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posX,
                      theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posY,
                      theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posZ, 1.0);
    rad = constRad;
#else // INTERLEAVED
    inPosition = vec4(thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posX,
                      thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posY,
                      thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posZ, 1.0);
    rad = constRad;
#endif // INTERLEAVED

#endif // VERTDATA_FLOAT_XYZ


#ifdef VERTDATA_DOUBLE_XYZ

#ifdef INTERLEAVED
    uvec2 thex = uvec2(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posX1,
                       theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posX2);
    uvec2 they = uvec2(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posY1,
                       theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posY2);
    uvec2 thez = uvec2(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posZ1,
                       theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posZ2);
    inPosition = vec4(float(packDouble2x32(thex)), float(packDouble2x32(they)),
                      float(packDouble2x32(thez)), 1.0);
    rad = constRad;
#else // INTERLEAVED
    uvec2 thex = uvec2(thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posX1,
                       thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posX2);
    uvec2 they = uvec2(thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posY1,
                       thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posY2);
    uvec2 thez = uvec2(thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posZ1,
                       thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posZ2);
    inPosition = vec4(float(packDouble2x32(thex)), float(packDouble2x32(they)),
                      float(packDouble2x32(thez)), 1.0);
    rad = constRad;
#endif // INTERLEAVED

#endif // VERTDATA_DOUBLE_XYZ


#ifdef VERTDATA_FLOAT_XYZR

#ifdef INTERLEAVED
    inPosition = vec4(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posX,
                      theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posY,
                      theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posZ, 1.0);
    rad = theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posR;
#else // INTERLEAVED
    inPosition = vec4(thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posX,
                      thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posY,
                      thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posZ, 1.0);
    rad = thePosBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].posR;
#endif // INTERLEAVED

#endif // VERTDATA_FLOAT_XYZR


#ifdef VERTDATA_DEFAULT
#endif // VERTDATA_DEFAULT

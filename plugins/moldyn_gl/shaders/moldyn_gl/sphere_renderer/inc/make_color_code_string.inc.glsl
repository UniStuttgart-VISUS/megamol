#ifdef COLDATA_NONE
    inColor = globalCol;
#endif // VERTDATA_NONE


#ifdef COLDATA_UINT8_RGB
    // false
#endif // COLDATA_UINT8_RGB


#ifdef COLDATA_UINT8_RGBA

#ifdef INTERLEAVED
    inColor = unpackUnorm4x8(theBuffer[SSBO_GENERATED_SHADER_INSTANCE + instanceOffset].color);
#else // INTERLEAVED
    inColor = unpackUnorm4x8(theColBuffer[SSBO_GENERATED_SHADER_INSTANCE + instanceOffset].color);
#endif // INTERLEAVED

#endif // COLDATA_UINT8_RGBA


#ifdef COLDATA_FLOAT_RGB

#ifdef INTERLEAVED
    inColor = vec4(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].r,
                   theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].g,
                   theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].b, 1.0);
#else // INTERLEAVED
    inColor = vec4(theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].r,
                   theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].g,
                   theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].b, 1.0);
#endif // INTERLEAVED

#endif // COLDATA_FLOAT_RGB


#ifdef COLDATA_FLOAT_RGBA

#ifdef INTERLEAVED
    inColor = vec4(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].r,
                   theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].g,
                   theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].b,
                   theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].a);
#else // INTERLEAVED
    inColor = vec4(theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].r,
                   theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].g,
                   theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].b,
                   theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].a);
#endif // INTERLEAVED

#endif // COLDATA_FLOAT_RGBA


#ifdef COLDATA_FLOAT_I

#ifdef INTERLEAVED
    inColIdx = theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].colorIndex;
#else // INTERLEAVED
    inColIdx = theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].colorIndex;
#endif // INTERLEAVED

#endif // COLDATA_FLOAT_I


#ifdef COLDATA_DOUBLE_I

#ifdef INTERLEAVED
    inColIdx = float(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].colorIndex);
#else // INTERLEAVED
    inColIdx = float(theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].colorIndex);
#endif // INTERLEAVED

#endif // COLDATA_DOUBLE_I


#ifdef COLDATA_USHORT_RGBA

#ifdef INTERLEAVED
    inColor.xy = unpackUnorm2x16(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].col1);
    inColor.zw = unpackUnorm2x16(theBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].col2);
#else // INTERLEAVED
    inColor.xy = unpackUnorm2x16(theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].col1);
    inColor.zw = unpackUnorm2x16(theColBuffer[SSBO_GENERATED_SHADER_INSTANCE  + instanceOffset].col2);
#endif // INTERLEAVED

#endif // COLDATA_USHORT_RGBA


#ifdef COLDATA_DEFAULT
    inColor = globalCol;
#endif // COLDATA_DEFAULT

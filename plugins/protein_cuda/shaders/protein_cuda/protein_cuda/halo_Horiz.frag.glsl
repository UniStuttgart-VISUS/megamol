#version 120

#define GAUSS_3x3
#undef GAUSS_3x3

//#define VERTICAL_FILTER

uniform sampler2D sourceTex;
uniform vec2 screenResInv;

#ifdef GAUSS_3x3
#   define FILTERSAMPLES 3
#   define FILTEROFFSET  1
    float gaussTable[3] = float[3]( 0.25, 0.5, 0.25 );
#else
#   define FILTERSAMPLES 5
#   define FILTEROFFSET  2
    float gaussTable[5] = float[5]( 2.0/30.0, 7.0/30.0, 12.0/30.0, 7.0/30.0, 2.0/30.0 );
#endif



void main(void)
{
    vec4 cOut = vec4(0.0);
    vec4 src = texture2D(sourceTex, gl_TexCoord[0].xy);

    vec2 stepSize = screenResInv;
#ifdef VERTICAL_FILTER
    stepSize.x = 0.0;
#else
    stepSize.y = 0.0;
#endif

    for (int i=0; i<FILTERSAMPLES; ++i)
    {
        vec2 texPos = gl_TexCoord[0].xy + stepSize * (i-FILTEROFFSET);
        vec4 sample = texture2D(sourceTex, texPos);
        cOut += gaussTable[i] * sample;
    }

    gl_FragColor =  cOut;
}

vec2 SampleBlurred( vec4 inPos, vec2 coord )
{
    float packedEdges = texelFetch(g_BlurInput, ivec2( inPos.xy), 0 ).y;
    vec4 edgesLRTB    = UnpackEdges( packedEdges );

float ssaoValue;
float ssaoValueL;
float ssaoValueT;
float ssaoValueR;
float ssaoValueB;

if(false) {
    // automatically done in our shader
    vec4 valuesUL = textureGather( g_BlurInput, coord );
    vec4 valuesBR = textureGather( g_BlurInput, coord );
    // vec4 valuesUL = textureGather( g_BlurInput, vec2(coord.x - g_ASSAOConsts.HalfViewportPixelSize.x * 0.5, coord.y + g_ASSAOConsts.HalfViewportPixelSize.y * 0.5) );
    // vec4 valuesBR = textureGather( g_BlurInput, vec2(coord.x + g_ASSAOConsts.HalfViewportPixelSize.x * 0.5, coord.y - g_ASSAOConsts.HalfViewportPixelSize.y * 0.5) );

    // fetch all ssaoValues around current pixel
    ssaoValue     = valuesUL.z;
    ssaoValueL    = valuesUL.w;
    ssaoValueT    = valuesUL.y;
    ssaoValueR    = valuesBR.y;
    ssaoValueB    = valuesBR.w;
} else {

    // THIS IS THE DEFAULT!
    // automatically done in our shader
    vec4 valuesBL = textureGather( g_BlurInput, coord - g_ASSAOConsts.HalfViewportPixelSize * 0.5 );
    vec4 valuesUR = textureGather( g_BlurInput, coord + g_ASSAOConsts.HalfViewportPixelSize * 0.5 );

    // fetch all ssaoValues around current pixel
    ssaoValue     = valuesBL.y;   // center   e.g. (5,5)                                          vUR.x
    ssaoValueL    = valuesBL.x;   // left     --> (4,5)                                   vBL.x   vBL.y   vUR.z
    ssaoValueT    = valuesUR.x;   // top      --> (5,6)                                           vBL.z
    ssaoValueR    = valuesUR.z;   // right    valuesBR.z == (6,6) --> .z = (6,5)
    ssaoValueB    = valuesBL.z;   // bottom   --> (5,4)
}

    float sumWeight = 0.5f;
    float sum = ssaoValue * sumWeight;

    AddSample( ssaoValueL, edgesLRTB.x, sum, sumWeight );
    AddSample( ssaoValueR, edgesLRTB.y, sum, sumWeight );

    AddSample( ssaoValueT, edgesLRTB.z, sum, sumWeight );
    AddSample( ssaoValueB, edgesLRTB.w, sum, sumWeight );

    float ssaoAvg = sum / sumWeight;

    ssaoValue = ssaoAvg; //min( ssaoValue, ssaoAvg ) * 0.2 + ssaoAvg * 0.8;

    return vec2( ssaoValue, packedEdges );
}

vec2 SampleBlurred( vec4 inPos, vec2 coord )
{
    float packedEdges   = g_BlurInput.Load( ivec3( inPos.xy, 0 ) ).y;
    vec4 edgesLRTB    = UnpackEdges( packedEdges );

                                                                                // automatically done in our shader
    vec4 valuesUL     = g_BlurInput.GatherRed( g_PointMirrorSampler, coord - g_ASSAOConsts.HalfViewportPixelSize * 0.5 );
    vec4 valuesBR     = g_BlurInput.GatherRed( g_PointMirrorSampler, coord + g_ASSAOConsts.HalfViewportPixelSize * 0.5 );

    // fetch all ssaoValues around current pixel
    float ssaoValue     = valuesUL.y;   // center   e.g. (5,5)                                          vUL.z
    float ssaoValueL    = valuesUL.x;   // left     --> (4,5)                                   vUL.x   vUL.y   vBR.z
    float ssaoValueT    = valuesUL.z;   // top      --> (5,4)                                           vBR.x   vBR.y
    float ssaoValueR    = valuesBR.z;   // right    valuesBR.z == (6,6) --> .z = (6,5)
    float ssaoValueB    = valuesBR.x;   // bottom   --> (5,6)

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

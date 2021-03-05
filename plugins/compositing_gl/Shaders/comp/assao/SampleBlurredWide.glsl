vec2 SampleBlurredWide( vec4 inPos, vec2 coord )
{
    vec2 vC           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2( 0,  0 ) ).xy;
    vec2 vL           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2( -2, 0 ) ).xy;
    vec2 vT           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2( 0, -2 ) ).xy;
    vec2 vR           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2(  2, 0 ) ).xy;
    vec2 vB           = g_BlurInput.SampleLevel( g_PointMirrorSampler, coord, 0.0, ivec2( 0,  2 ) ).xy;

    float packedEdges   = vC.y;
    vec4 edgesLRTB    = UnpackEdges( packedEdges );
    edgesLRTB.x         *= UnpackEdges( vL.y ).y;
    edgesLRTB.z         *= UnpackEdges( vT.y ).w;
    edgesLRTB.y         *= UnpackEdges( vR.y ).x;
    edgesLRTB.w         *= UnpackEdges( vB.y ).z;

    float ssaoValue     = vC.x;
    float ssaoValueL    = vL.x;
    float ssaoValueT    = vT.x;
    float ssaoValueR    = vR.x;
    float ssaoValueB    = vB.x;

    float sumWeight = 0.8f;
    float sum = ssaoValue * sumWeight;

    AddSample( ssaoValueL, edgesLRTB.x, sum, sumWeight );
    AddSample( ssaoValueR, edgesLRTB.y, sum, sumWeight );
    AddSample( ssaoValueT, edgesLRTB.z, sum, sumWeight );
    AddSample( ssaoValueB, edgesLRTB.w, sum, sumWeight );

    float ssaoAvg = sum / sumWeight;

    ssaoValue = ssaoAvg; //min( ssaoValue, ssaoAvg ) * 0.2 + ssaoAvg * 0.8;

    return vec2( ssaoValue, packedEdges );
}

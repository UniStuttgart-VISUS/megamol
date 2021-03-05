void PSPrepareDepths( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
{
#if 0   // gather can be a bit faster but doesn't work with input depth buffers that don't match the working viewport
    vec2 gatherUV = inPos.xy * g_ASSAOConsts.Viewport2xPixelSize;
    vec4 depths = g_DepthSource.GatherRed( g_PointClampSampler, gatherUV );
    float a = depths.w;  // g_DepthSource.Load( ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 0, 0 ) ).x;
    float b = depths.z;  // g_DepthSource.Load( ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 1, 0 ) ).x;
    float c = depths.x;  // g_DepthSource.Load( ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 0, 1 ) ).x;
    float d = depths.y;  // g_DepthSource.Load( ivec3( ivec2(inPos.xy) * 2, 0 ), ivec2( 1, 1 ) ).x;
#else
    ivec3 baseCoord = ivec3( ivec2(inPos.xy) * 2, 0 );
    float a = g_DepthSource.Load( baseCoord, ivec2( 0, 0 ) ).x;
    float b = g_DepthSource.Load( baseCoord, ivec2( 1, 0 ) ).x;
    float c = g_DepthSource.Load( baseCoord, ivec2( 0, 1 ) ).x;
    float d = g_DepthSource.Load( baseCoord, ivec2( 1, 1 ) ).x;
#endif

    out0 = ScreenSpaceToViewSpaceDepth( a );
    out1 = ScreenSpaceToViewSpaceDepth( b );
    out2 = ScreenSpaceToViewSpaceDepth( c );
    out3 = ScreenSpaceToViewSpaceDepth( d );
}

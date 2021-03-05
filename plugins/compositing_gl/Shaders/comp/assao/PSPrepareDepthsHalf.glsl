void PSPrepareDepthsHalf( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1 )
{
    ivec3 baseCoord = ivec3( ivec2(inPos.xy) * 2, 0 );
    float a = g_DepthSource.Load( baseCoord, ivec2( 0, 0 ) ).x;
    float d = g_DepthSource.Load( baseCoord, ivec2( 1, 1 ) ).x;

    out0 = ScreenSpaceToViewSpaceDepth( a );
    out1 = ScreenSpaceToViewSpaceDepth( d );
}

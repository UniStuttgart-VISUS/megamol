layout(local_size_x = 8, local_size_y = 8) in;

//void PSPrepareDepthsHalf( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1 )
void main()
{
    vec3 inPos = gl_GlobalInvocationID;

    ivec3 baseCoord = ivec3( ivec2(inPos.xy) * 2, 0 );
    float a = texelFetchOffset(g_DepthSource, baseCoord.xy, 0, ivec2( 0, 0 ) ).x;
    //float a = g_DepthSource.Load( baseCoord, ivec2( 0, 0 ) ).x;
    float d = texelFetchOffset(g_DepthSource, baseCoord.xy, 0, ivec2( 1, 1 ) ).x;
    //float d = g_DepthSource.Load( baseCoord, ivec2( 1, 1 ) ).x;

    float out0 = ScreenSpaceToViewSpaceDepth( a );
    float out3 = ScreenSpaceToViewSpaceDepth( d );

    imageStore(g_HalfDepths0, ivec2(inPos.xy), vec4(out0, 0.f, 0.f, 0.f));
    imageStore(g_HalfDepths3, ivec2(inPos.xy), vec4(out3, 0.f, 0.f, 0.f));
}

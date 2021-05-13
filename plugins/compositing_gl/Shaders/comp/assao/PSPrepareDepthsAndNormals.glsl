layout(local_size_x = 8, local_size_y = 8) in;

//void PSPrepareDepthsAndNormals( in vec4 inPos : SV_POSITION, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
void main()
{
    vec3 inPos = gl_GlobalInvocationID;

    ivec2 baseCoords = (( ivec2 )inPos.xy) * 2;
    vec2  bottomLeftUV = (inPos.xy - 0.25) * g_ASSAOConsts.Viewport2xPixelSize;

#if 0   // gather can be a bit faster but doesn't work with input depth buffers that don't match the working viewport
    vec2 gatherUV = inPos.xy * g_ASSAOConsts.Viewport2xPixelSize;
    vec4 depths = textureGather(g_DepthSource, gatherUV );
    float out0 = ScreenSpaceToViewSpaceDepth( depths.x );
    float out1 = ScreenSpaceToViewSpaceDepth( depths.y );
    float out2 = ScreenSpaceToViewSpaceDepth( depths.w );
    float out3 = ScreenSpaceToViewSpaceDepth( depths.z );
#else
    ivec3 baseCoord = ivec3( ivec2(inPos.xy) * 2, 0 );
    float out0 = ScreenSpaceToViewSpaceDepth( texelFetchOffset(g_DepthSource, baseCoord, ivec2( 0, 1 ) ).x );
    float out1 = ScreenSpaceToViewSpaceDepth( texelFetchOffset(g_DepthSource, baseCoord, ivec2( 1, 1 ) ).x );
    float out2 = ScreenSpaceToViewSpaceDepth( texelFetchOffset(g_DepthSource, baseCoord, ivec2( 0, 0 ) ).x );
    float out3 = ScreenSpaceToViewSpaceDepth( texelFetchOffset(g_DepthSource, baseCoord, ivec2( 1, 0 ) ).x );
#endif

    float pixZs[4][4];

    // CAUTION: converting from hlsl to glsl could be error-prone
    // it is highly possible that something is wrong here
    // middle 4
    pixZs[1][1] = out2;
    pixZs[2][1] = out3;
    pixZs[1][2] = out0;
    pixZs[2][2] = out1;
    // left 2
    pixZs[0][1] = ScreenSpaceToViewSpaceDepth(  textureLodOffset(g_DepthSource, bottomLeftUV, 0.0, ivec2( -1, 0 ) ).x );
    pixZs[0][2] = ScreenSpaceToViewSpaceDepth(  textureLodOffset(g_DepthSource, bottomLeftUV, 0.0, ivec2( -1, 1 ) ).x );
    // right 2
    pixZs[3][1] = ScreenSpaceToViewSpaceDepth(  textureLodOffset(g_DepthSource, bottomLeftUV, 0.0, ivec2(  2, 0 ) ).x );
    pixZs[3][2] = ScreenSpaceToViewSpaceDepth(  textureLodOffset(g_DepthSource, bottomLeftUV, 0.0, ivec2(  2, 1 ) ).x );
    // TODO: currently top and bottom is flipped, but is this necessary?
    // top 2
    pixZs[1][3] = ScreenSpaceToViewSpaceDepth(  textureLodOffset(g_DepthSource, bottomLeftUV, 0.0, ivec2(  0, -1 ) ).x );
    pixZs[2][3] = ScreenSpaceToViewSpaceDepth(  textureLodOffset(g_DepthSource, bottomLeftUV, 0.0, ivec2(  1, -1 ) ).x );
    // bottom 2
    pixZs[1][0] = ScreenSpaceToViewSpaceDepth(  textureLodOffset(g_DepthSource, bottomLeftUV, 0.0, ivec2(  0,  2 ) ).x );
    pixZs[2][0] = ScreenSpaceToViewSpaceDepth(  textureLodOffset(g_DepthSource, bottomLeftUV, 0.0, ivec2(  1,  2 ) ).x );

    vec4 edges0 = CalculateEdges( pixZs[1][1], pixZs[0][1], pixZs[2][1], pixZs[1][0], pixZs[1][2] );
    vec4 edges1 = CalculateEdges( pixZs[2][1], pixZs[1][1], pixZs[3][1], pixZs[2][0], pixZs[2][2] );
    vec4 edges2 = CalculateEdges( pixZs[1][2], pixZs[0][2], pixZs[2][2], pixZs[1][1], pixZs[1][3] );
    vec4 edges3 = CalculateEdges( pixZs[2][2], pixZs[1][2], pixZs[3][2], pixZs[2][1], pixZs[2][3] );

    vec3 pixPos[4][4];
    // middle 4
    pixPos[1][1] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0,  0.0 ), pixZs[1][1] );
    pixPos[2][1] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0,  0.0 ), pixZs[2][1] );
    pixPos[1][2] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0,  1.0 ), pixZs[1][2] );
    pixPos[2][2] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0,  1.0 ), pixZs[2][2] );
    // left 2
    pixPos[0][1] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( -1.0,  0.0), pixZs[0][1] );
    pixPos[0][2] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( -1.0,  1.0), pixZs[0][2] );
    // right 2
    pixPos[3][1] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2(  2.0,  0.0), pixZs[3][1] );
    pixPos[3][2] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2(  2.0,  1.0), pixZs[3][2] );
    // top 2
    pixPos[1][3] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0, -1.0 ), pixZs[1][0] );
    pixPos[2][3] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0, -1.0 ), pixZs[2][0] );
    // bottom 2
    pixPos[1][0] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 0.0,  2.0 ), pixZs[1][3] );
    pixPos[2][0] = NDCToViewspace( bottomLeftUV + g_ASSAOConsts.ViewportPixelSize * vec2( 1.0,  2.0 ), pixZs[2][3] );

    vec3 norm0 = CalculateNormal( edges0, pixPos[1][1], pixPos[0][1], pixPos[2][1], pixPos[1][0], pixPos[1][2] );
    vec3 norm1 = CalculateNormal( edges1, pixPos[2][1], pixPos[1][1], pixPos[3][1], pixPos[2][0], pixPos[2][2] );
    vec3 norm2 = CalculateNormal( edges2, pixPos[1][2], pixPos[0][2], pixPos[2][2], pixPos[1][1], pixPos[1][3] );
    vec3 norm3 = CalculateNormal( edges3, pixPos[2][2], pixPos[1][2], pixPos[3][2], pixPos[2][1], pixPos[2][3] );

    imageStore(g_NormalsOutputUAV, baseCoords + ivec2( 0, 0 ), vec4( norm0 * 0.5 + 0.5, 0.0 ));
    imageStore(g_NormalsOutputUAV, baseCoords + ivec2( 1, 0 ), vec4( norm1 * 0.5 + 0.5, 0.0 ));
    imageStore(g_NormalsOutputUAV, baseCoords + ivec2( 0, 1 ), vec4( norm2 * 0.5 + 0.5, 0.0 ));
    imageStore(g_NormalsOutputUAV, baseCoords + ivec2( 1, 1 ), vec4( norm3 * 0.5 + 0.5, 0.0 ));

    imageStore(g_HalfDepthsMipView0, inPos.xy, vec4( out0, 0.f, 0.f, 0.f ));
    imageStore(g_HalfDepthsMipView1, inPos.xy, vec4( out1, 0.f, 0.f, 0.f ));
    imageStore(g_HalfDepthsMipView2, inPos.xy, vec4( out2, 0.f, 0.f, 0.f ));
    imageStore(g_HalfDepthsMipView3, inPos.xy, vec4( out3, 0.f, 0.f, 0.f ));

    //g_NormalsOutputUAV[ baseCoords + ivec2( 0, 0 ) ] = vec4( norm0 * 0.5 + 0.5, 0.0 );
    //g_NormalsOutputUAV[ baseCoords + ivec2( 1, 0 ) ] = vec4( norm1 * 0.5 + 0.5, 0.0 );
    //g_NormalsOutputUAV[ baseCoords + ivec2( 0, 1 ) ] = vec4( norm2 * 0.5 + 0.5, 0.0 );
    //g_NormalsOutputUAV[ baseCoords + ivec2( 1, 1 ) ] = vec4( norm3 * 0.5 + 0.5, 0.0 );
}

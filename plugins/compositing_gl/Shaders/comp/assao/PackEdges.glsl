// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
float PackEdges( vec4 edgesLRTB )
{
//    ivec4 edgesLRTBi = ivec4( saturate( edgesLRTB ) * 3.0 + 0.5 );
//    return ( (edgesLRTBi.x << 6) + (edgesLRTBi.y << 4) + (edgesLRTBi.z << 2) + (edgesLRTBi.w << 0) ) / 255.0;

    // optimized, should be same as above
    edgesLRTB = round( clamp( edgesLRTB, 0.0, 1.0 ) ) * 3.05 );
    return dot( edgesLRTB, vec4( 64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0 ) ) ;
}

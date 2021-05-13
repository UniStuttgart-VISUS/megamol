layout(local_size_x = 8, local_size_y = 8) in;

//void PSGenerateQ0( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out vec2 out0 : SV_Target0 )
void main()
{
    vec3 inPos = gl_GlobalInvocationID;

    float outShadowTerm;
    float outWeight;
    vec4  outEdges;
    GenerateSSAOShadowsInternal( outShadowTerm, outEdges, outWeight, inPos.xy/*, inUV*/, 0, false );

    vec2 out0(0.f);
    out0.x = outShadowTerm;
    out0.y = PackEdges( vec4( 1, 1, 1, 1 ) ); // no edges in low quality

    imageStore(g_PingPongHalfResultA, inPos.xy, vec4(out0, 0.f, 0.f));
}

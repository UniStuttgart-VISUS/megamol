void PSGenerateQ0( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out vec2 out0 : SV_Target0 )
{
    float   outShadowTerm;
    float   outWeight;
    vec4  outEdges;
    GenerateSSAOShadowsInternal( outShadowTerm, outEdges, outWeight, inPos.xy/*, inUV*/, 0, false );
    out0.x = outShadowTerm;
    out0.y = PackEdges( vec4( 1, 1, 1, 1 ) ); // no edges in low quality
}

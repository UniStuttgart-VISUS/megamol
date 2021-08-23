layout(local_size_x = 8, local_size_y = 8) in;

//void PSGenerateQ2( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out vec2 out0 : SV_Target0 )
void main()
{
    vec3 inPos = gl_GlobalInvocationID;

    float outShadowTerm;
    float outWeight;
    vec4  outEdges;
	vec3 normal;
    GenerateSSAOShadowsInternal( outShadowTerm, outEdges, outWeight, inPos.xy/*, inUV*/, 2, false, normal );

    vec2 out0 = vec2(0.f);
    out0.x = outShadowTerm;
    out0.y = PackEdges( outEdges );

    imageStore(g_PingPongHalfResultA, ivec2(inPos.xy), vec4(out0, 0.f, 0.f));
}

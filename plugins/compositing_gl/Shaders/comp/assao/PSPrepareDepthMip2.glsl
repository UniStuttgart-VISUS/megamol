layout(local_size_x = 8, local_size_y = 8) in;

//void PSPrepareDepthMip1( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
void main()
{
    vec4 inPos = vec4(gl_GlobalInvocationID, 0.f);

    float out0 = 0.f;
    float out1 = 0.f;
    float out2 = 0.f;
    float out3 = 0.f;

    PrepareDepthMip( inPos/*, inUV*/, 2, out0, out1, out2, out3 );

    imageStore(g_HalfDepthsMipView0, ivec2(inPos.xy), vec4(out0, 0.f, 0.f, 0.f));
    imageStore(g_HalfDepthsMipView1, ivec2(inPos.xy), vec4(out1, 0.f, 0.f, 0.f));
    imageStore(g_HalfDepthsMipView2, ivec2(inPos.xy), vec4(out2, 0.f, 0.f, 0.f));
    imageStore(g_HalfDepthsMipView3, ivec2(inPos.xy), vec4(out3, 0.f, 0.f, 0.f));
}

void PSPrepareDepthMip3( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/, out float out0 : SV_Target0, out float out1 : SV_Target1, out float out2 : SV_Target2, out float out3 : SV_Target3 )
{
    PrepareDepthMip( inPos/*, inUV*/, 3, out0, out1, out2, out3 );
}

// edge-sensitive blur (wider kernel)
//vec2 PSSmartBlurWide( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
void main()
{
    vec3 inPos = gl_GlobalInvocationID;
    vec3 viewport = imageSize(g_PingPongHalfResultA);
    vec2 inUV = (2.f * inPos.xy + vec2(1.f)) / (2.f * vec2(viewport.xy));

    vec2 returnVal = SampleBlurredWide( vec4(inPos, 1.f), inUV );

    imageStore(g_PingPongHalfResultB, inPos.xy, vec4(returnVal, 0.f, 0.f));
}

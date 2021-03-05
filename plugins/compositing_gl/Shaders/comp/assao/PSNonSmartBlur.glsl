// edge-ignorant blur in x and y directions, 9 pixels touched (for the lowest quality level 0)
vec2 PSNonSmartBlur( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    vec2 halfPixel = g_ASSAOConsts.HalfViewportPixelSize * 0.5f;

    vec2 centre = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV, 0.0 ).xy;

    vec4 vals;
    vals.x = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV + vec2( -halfPixel.x * 3, -halfPixel.y ), 0.0 ).x;
    vals.y = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV + vec2( +halfPixel.x, -halfPixel.y * 3 ), 0.0 ).x;
    vals.z = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV + vec2( -halfPixel.x, +halfPixel.y * 3 ), 0.0 ).x;
    vals.w = g_BlurInput.SampleLevel( g_LinearClampSampler, inUV + vec2( +halfPixel.x * 3, +halfPixel.y ), 0.0 ).x;

    return vec2(dot( vals, 0.2.xxxx ) + centre.x * 0.2, centre.y);
}

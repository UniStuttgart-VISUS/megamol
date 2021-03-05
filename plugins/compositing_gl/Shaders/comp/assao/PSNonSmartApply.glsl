// edge-ignorant blur & apply (for the lowest quality level 0)
vec4 PSNonSmartApply( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    float a = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 0 ), 0.0 ).x;
    float b = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 1 ), 0.0 ).x;
    float c = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 2 ), 0.0 ).x;
    float d = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 3 ), 0.0 ).x;
    float avg = (a+b+c+d) * 0.25;
    return vec4( avg.xxx, 1.0 );
}

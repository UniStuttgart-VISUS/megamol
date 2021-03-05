// edge-ignorant blur & apply, skipping half pixels in checkerboard pattern (for the Lowest quality level 0 and Settings::SkipHalfPixelsOnLowQualityLevel == true )
vec4 PSNonSmartHalfApply( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    float a = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 0 ), 0.0 ).x;
    float d = g_FinalSSAO.SampleLevel( g_LinearClampSampler, vec3( inUV.xy, 3 ), 0.0 ).x;
    float avg = (a+d) * 0.5;
    return vec4( avg.xxx, 1.0 );
}

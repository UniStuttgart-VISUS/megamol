layout(local_size_x = 8, local_size_y = 8) in;

// edge-ignorant blur & apply, skipping half pixels in checkerboard pattern (for the Lowest quality level 0 and Settings::SkipHalfPixelsOnLowQualityLevel == true )
//vec4 PSNonSmartHalfApply( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
void main()
{
    vec3 inPos = gl_GlobalInvocationID;
    vec3 viewport = imageSize(g_FinalSSAO);
    vec2 inUV = (2.f * inPos.xy + vec2(1.f)) / (2.f * vec2(viewport.xy));

    float a = textureLod(g_FinalSSAO, vec3( inUV, 0 ), 0.0 ).x;
    float d = textureLod(g_FinalSSAO, vec3( inUV, 3 ), 0.0 ).x;
    float avg = (a+d) * 0.5;

    //return vec4( avg.xxx, 1.0 );
    imageStore(g_FinalOutput, inPos.xy, vec4(avg.xxx, 1.f));
}

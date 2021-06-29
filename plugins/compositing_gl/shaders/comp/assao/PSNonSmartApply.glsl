layout(local_size_x = 8, local_size_y = 8) in;

// edge-ignorant blur & apply (for the lowest quality level 0)
//vec4 PSNonSmartApply( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
void main()
{
    vec3 inPos = gl_GlobalInvocationID;
    ivec3 viewport = textureSize(g_FinalSSAO, 0);
    vec2 inUV = (2.f * inPos.xy + vec2(1.f)) / (2.f * vec2(viewport.xy));

    // TODO: look into the host code: is this the correct sampler?
    // TODO: is the z-coordinate in vec3(inUV, X) correct?
    // I would suspect you would need texture coordinate there as well
    float a = textureLod(g_FinalSSAO, vec3( inUV, 0 ), 0.0 ).x;
    float b = textureLod(g_FinalSSAO, vec3( inUV, 1 ), 0.0 ).x;
    float c = textureLod(g_FinalSSAO, vec3( inUV, 2 ), 0.0 ).x;
    float d = textureLod(g_FinalSSAO, vec3( inUV, 3 ), 0.0 ).x;
    float avg = (a+b+c+d) * 0.25;

    //return vec4( avg.xxx, 1.0 );
    imageStore(g_FinalOutput, ivec2(inPos.xy), vec4(avg.xxx, 1.f));
}

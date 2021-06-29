layout(local_size_x = 8, local_size_y = 8) in;

// edge-ignorant blur in x and y directions, 9 pixels touched (for the lowest quality level 0)
//vec2 PSNonSmartBlur( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
void main()
{
    vec3 inPos = gl_GlobalInvocationID;
    ivec2 viewport = imageSize(g_PingPongHalfResultB);
    // TODO: is this correct? error-prone
    vec2 inUV = (2.f * inPos.xy + vec2(1.f)) / (2.f * vec2(viewport));

    vec2 halfPixel = g_ASSAOConsts.HalfViewportPixelSize * 0.5f;

    vec2 centre = textureLod(g_BlurInput, inUV, 0.0 ).xy;

    vec4 vals;
    vals.x = textureLod(g_BlurInput, inUV + vec2( -halfPixel.x * 3, -halfPixel.y ), 0.0 ).x;
    vals.y = textureLod(g_BlurInput, inUV + vec2( +halfPixel.x, -halfPixel.y * 3 ), 0.0 ).x;
    vals.z = textureLod(g_BlurInput, inUV + vec2( -halfPixel.x, +halfPixel.y * 3 ), 0.0 ).x;
    vals.w = textureLod(g_BlurInput, inUV + vec2( +halfPixel.x * 3, +halfPixel.y ), 0.0 ).x;

    vec2 returnVal = vec2(dot( vals, 0.2.xxxx ) + centre.x * 0.2, centre.y);
    //return vec2(dot( vals, 0.2.xxxx ) + centre.x * 0.2, centre.y);
    imageStore(g_PingPongHalfResultB, ivec2(inPos.xy), vec4(returnVal, 0.f, 0.f));
}

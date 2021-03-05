// edge-sensitive blur (wider kernel)
vec2 PSSmartBlurWide( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    return SampleBlurredWide( inPos, inUV );
}

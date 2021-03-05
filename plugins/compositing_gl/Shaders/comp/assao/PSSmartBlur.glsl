// edge-sensitive blur
vec2 PSSmartBlur( in vec4 inPos : SV_POSITION, in vec2 inUV : TEXCOORD0 ) : SV_Target
{
    return SampleBlurred( inPos, inUV );
}

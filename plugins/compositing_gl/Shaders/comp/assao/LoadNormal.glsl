vec3 LoadNormal( ivec2 pos )
{
    //vec3 encodedNormal = g_NormalmapSource.Load( ivec3( pos, 0 ) ).xyz;
    vec3 encodedNormal = texelFetch(g_NormalmapSource, ivec3( pos, 0 ) ).xyz;
    return DecodeNormal( encodedNormal );
}

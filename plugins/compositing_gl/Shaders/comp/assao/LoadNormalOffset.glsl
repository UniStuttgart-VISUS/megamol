vec3 LoadNormal( ivec2 pos, ivec2 offset )
{
    //vec3 encodedNormal = g_NormalmapSource.Load( ivec3( pos, 0 ), offset ).xyz;
    vec3 encodedNormal = texelFetch(g_NormalmapSource, ivec3( pos, 0 ), offset ).xyz;
    return DecodeNormal( encodedNormal );
}

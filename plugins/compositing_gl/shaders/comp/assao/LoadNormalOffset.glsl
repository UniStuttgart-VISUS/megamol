vec3 LoadNormal( ivec2 pos, ivec2 offset )
{
    //vec3 encodedNormal = g_NormalmapSource.Load( ivec3( pos, 0 ), offset ).xyz;
    vec3 encodedNormal = texelFetchOffset(g_NormalmapSource, pos, 0, offset ).xyz;
    encodedNormal = transpose(inverse(mat3(g_ASSAOConsts.viewMX))) * (encodedNormal);
    //return DecodeNormal( encodedNormal );
    return normalize(encodedNormal);
}

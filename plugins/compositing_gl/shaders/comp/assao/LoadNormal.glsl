vec3 LoadNormal( ivec2 pos )
{
    //vec3 encodedNormal = g_NormalmapSource.Load( ivec3( pos, 0 ) ).xyz;
    vec3 encodedNormal = texelFetch(g_NormalmapSource, pos, 0 ).xyz;
    // quick hack for normals, implement this in decode normals or ask here if world to view conversion is set
	encodedNormal = transpose(inverse(mat3(g_ASSAOConsts.viewMX))) * (encodedNormal);
    //return DecodeNormal( encodedNormal );
    return normalize(encodedNormal);
}

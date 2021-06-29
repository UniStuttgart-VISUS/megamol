vec3 DecodeNormal( vec3 encodedNormal )
{
    vec3 normal = encodedNormal * g_ASSAOConsts.NormalsUnpackMul.xxx + g_ASSAOConsts.NormalsUnpackAdd.xxx;

#if SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
    normal = mul( normal, (mat3)g_ASSAOConsts.NormalsWorldToViewspaceMatrix ).xyz;
#endif

    // normal = normalize( normal );    // normalize adds around 2.5% cost on High settings but makes little (PSNR 66.7) visual difference when normals are as in the sample (stored in R8G8B8A8_UNORM,
    //                                  // decoded in the shader), however it will likely be required if using different encoding/decoding or the inputs are not normalized, etc.

    return normal;
}

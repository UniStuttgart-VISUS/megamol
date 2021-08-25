vec3 ClipSpaceToViewSpacePosition( vec2 clipPos, float viewspaceDepth )
{
    // TODO: should not be correct atm
    return vec3( g_ASSAOConsts.CameraTanHalfFOV.xy * viewspaceDepth * clipPos, viewspaceDepth );
}

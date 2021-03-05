vec3 ClipSpaceToViewSpacePosition( vec2 clipPos, float viewspaceDepth )
{
    return vec3( g_ASSAOConsts.CameraTanHalfFOV.xy * viewspaceDepth * clipPos, viewspaceDepth );
}

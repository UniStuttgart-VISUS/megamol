vec3 ScreenSpaceToViewSpacePosition( vec2 screenPos, float viewspaceDepth )
{
    return vec3( g_ASSAOConsts.CameraTanHalfFOV.xy * viewspaceDepth * ScreenSpaceToClipSpacePositionXY( screenPos ), viewspaceDepth );
}

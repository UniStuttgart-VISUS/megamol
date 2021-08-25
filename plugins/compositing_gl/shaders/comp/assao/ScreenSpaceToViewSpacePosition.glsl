vec3 ScreenSpaceToViewSpacePosition( vec2 screenPos, float viewspaceDepth )
{
    // TODO: is this correct with the new math? probably not
    return vec3( g_ASSAOConsts.CameraTanHalfFOV.xy * viewspaceDepth * ScreenSpaceToClipSpacePositionXY( screenPos ), viewspaceDepth );
}

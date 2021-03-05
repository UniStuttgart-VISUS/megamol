// from [0, width], [0, height] to [-1, 1], [-1, 1]
vec2 ScreenSpaceToClipSpacePositionXY( vec2 screenPos )
{
    return screenPos * g_ASSAOConsts.Viewport2xPixelSize.xy - vec2( 1.0f, 1.0f );
}

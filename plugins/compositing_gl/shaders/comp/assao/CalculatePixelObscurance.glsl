float CalculatePixelObscurance( vec3 pixelNormal, vec3 hitDelta, float falloffCalcMulSq )
{
  float lengthSq = dot( hitDelta, hitDelta );
  float NdotD = dot(pixelNormal, hitDelta) / sqrt(lengthSq);

  float falloffMult = max( 0.0, lengthSq * falloffCalcMulSq + 1.0 );

  return max( 0, NdotD - g_ASSAOConsts.EffectHorizonAngleThreshold ) * falloffMult;
}

// *******************************************************
// Smoothing function for edges.
//
// Return value can be interpreted as alpha value.
//
// Expects 'distances' in the range 0.0 to 1.0.
// The edge is expected to lie at distance 0.5.
// Distances greater than 0.5 are interpreted as lying 'inside'.
// Distances less than 0.5 are interpreted as lying 'outside'.
// Distances of 0.0 are discarded.
// *******************************************************
float smoothing(const in float distance) {
    if (distance <= 0.0) {
        discard;
    }
    float dist = clamp(distance, 0.0, 1.0);
    float smootingEdge = 0.99 * length(vec2(dFdx(dist), dFdy(dist)));
    float smootingValue =  smoothstep((0.5 - smootingEdge), (0.5 + smootingEdge), dist);
    smootingValue = clamp(smootingValue, 0.0, 1.0);
    if (smootingValue == 0.0) {
        discard;
    }
    return smootingValue;
}

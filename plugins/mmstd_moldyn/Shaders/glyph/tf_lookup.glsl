vec4 tf_lookup(sampler1D tf_texture, vec2 tf_range, float value) {
    // Normalize value and clamp to [0.0;1.0].
    float normalizedValue = clamp((value - tf_range.x) / (tf_range.y - tf_range.x), 0.0, 1.0);
    // Do a texture lookup between the center of first texel center and 
    // the last texel to ensure proper interpolation.
    float dx = 1.0 / textureSize(tf_texture, 0);
    float uOffset = dx * 0.5;
    float uRange = 1.0 - dx;
    return texture(tf_texture, normalizedValue * uRange + uOffset);
}
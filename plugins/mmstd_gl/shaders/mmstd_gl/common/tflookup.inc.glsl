#ifndef MMSTD_GL_TFLOOKUP_INC_GLSL
#define MMSTD_GL_TFLOOKUP_INC_GLSL

// Does a color lookup for a non-normalized value using a transfer function.
// @see CallGetTransferFunction::TextureCoordinates()
// @see CallGetTransferFunction::Range()
vec4 tflookup(sampler1D tfTexture, vec2 tfRange, float value) {
    // Normalize value and clamp to [0.0;1.0].
    float normalizedValue = clamp((value - tfRange.x) / (tfRange.y - tfRange.x), 0.0, 1.0);
    // Do a texture lookup between the center of first texel center and
    // the last texel to ensure proper interpolation.
    float dx = 1.0 / textureSize(tfTexture, 0);
    float uOffset = dx * 0.5;
    float uRange = 1.0 - dx;
    return texture(tfTexture, normalizedValue * uRange + uOffset);
}

#endif // MMSTD_GL_TFLOOKUP_INC_GLSL

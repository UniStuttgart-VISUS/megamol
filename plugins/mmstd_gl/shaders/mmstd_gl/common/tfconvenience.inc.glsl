#ifndef MMSTD_GL_TFCONVENIENCE_INC_GLSL
#define MMSTD_GL_TFCONVENIENCE_INC_GLSL

uniform sampler1D tfTexture;
uniform vec2 tfRange;

// Convenience function for `tflookup()`.
vec4 tflookup(float value) {
    return tflookup(tfTexture, tfRange, value);
}

#endif // MMSTD_GL_TFCONVENIENCE_INC_GLSL

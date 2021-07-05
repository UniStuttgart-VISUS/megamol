#ifndef CORE_TFCONVENIENCE_INC_GLSL
#define CORE_TFCONVENIENCE_INC_GLSL

uniform sampler1D tfTexture;
uniform vec2 tfRange;

// Convenience function for `tflookup()`.
vec4 tflookup(float value) {
    return tflookup(tfTexture, tfRange, value);
}

#endif // CORE_TFCONVENIENCE_INC_GLSL

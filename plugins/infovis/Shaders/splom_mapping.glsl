uniform int valueMapping;

uniform int valueColumn;
uniform vec2 valueColumnMinMax;

uniform sampler1D colorTable;
uniform int colorCount;

uniform float alphaScaling;

// Normalize value to [0;1].
float normalizeValue(float value) {
   return  (value - valueColumnMinMax.x) / (valueColumnMinMax.y - valueColumnMinMax.x);
}

// Resolves a normalized value to the corresponding color.
vec4 normalizedValueToColor(float normalizedValue) {
    const float colorOffset = normalizedValue + 0.5 / float(colorCount);
    return texture(colorTable, colorOffset);
}

// Emits a vec4 for the screen shader, assuming regular alpha-blending for RGB
// and blending against white for A (yup, craziness).
vec4 toScreen(float value, vec4 valueColor, float density) {
    switch (valueMapping) {
    case 0:
        // Do a color lookup, then blend against one.
        return vec4(valueColor.rgb, valueColor.a * density * alphaScaling);
    case 1:
        // Do kernel density estimation, then do a color lookup (deferred)
        return vec4(1.0, 1.0, 1.0, density);
    case 2:
        // Weighted kernel density estimation, then do a color lookup (deferred).
        return vec4(value, value, value, density);
    default:
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
}

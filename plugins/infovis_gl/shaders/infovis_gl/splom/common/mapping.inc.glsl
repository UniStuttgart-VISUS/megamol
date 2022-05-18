uniform int valueMapping;

uniform int valueColumn;
uniform vec2 valueColumnMinMax;

uniform float alphaScaling;

uniform vec4 pickColor = vec4(1.0, 0.0, 0.0, 1.0);

// Normalize value to [0;1].
float normalizeValue(float value) {
    return (value - valueColumnMinMax.x) / (valueColumnMinMax.y - valueColumnMinMax.x);
}

// Adjusts the color based on flags.
vec4 flagifyColor(vec4 color, uint flags) {
    if (bitflag_test(flags, FLAG_SELECTED, FLAG_SELECTED)) {
        return pickColor;
    } else {
        return color;
    }
}

// Emits a vec4 for the screen shader, assuming regular alpha-blending for RGB
// and blending against white for A (yup, craziness).
vec4 toScreen(float value, vec4 valueColor, float density) {
    switch (valueMapping) {
    case 0:
        // Do a color lookup, then blend against one.
        return vec4(valueColor.rgb, valueColor.a * density);
    case 1:
        // Do kernel density estimation, then do a color lookup (deferred)
        return vec4(density, 0.0, 0.0, density > 0.0 ? 1.0 : 0.0);
    case 2:
        // Weighted kernel density estimation, then do a color lookup (deferred).
        return vec4(value * density, 0.0, 0.0, density > 0.0 ? 1.0 : 0.0);
    default:
        return vec4(0.0);
    }
}

// Unpacks the screen RGBA value and does the post-processing for toScreen().
vec4 fromScreen(vec4 screen) {
    switch (valueMapping) {
    case 0:
        return vec4(screen.rgb, screen.a * alphaScaling);
    case 1:
    case 2:
        //XXX: KDE without normalization to Unit-N or Sum(Weights) might not be fun.
        vec4 color = tflookup(screen.r);
        return vec4(color.rgb, color.a * screen.a * alphaScaling);
    default:
        return vec4(0.0);
    }
}

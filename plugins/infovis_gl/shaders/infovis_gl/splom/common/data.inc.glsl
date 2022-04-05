layout(std430, binding = 3) buffer ValueSSBO {
    float values[];
};

layout(std430, binding = 4) buffer FlagBuffer {
    uint flags[];
};

// Maps a pair of values to the position.
vec4 valuesToPosition(Plot plot, vec2 values) {
    vec2 unitValues = vec2((values.x - plot.minX) / (plot.maxX - plot.minX),
        (values.y - plot.minY) / (plot.maxY - plot.minY));
#if DEBUG_MINMAX
    // To debug column min/max and thus normalization, we clamp the result.
    unitValues = clamp(unitValues, vec2(0.0), vec2(1.0));
#endif
    return vec4(unitValues.x * plot.sizeX + plot.offsetX,
        unitValues.y * plot.sizeY + plot.offsetY,
        0.0, 1.0);
}

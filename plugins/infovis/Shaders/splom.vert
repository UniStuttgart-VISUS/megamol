//#define DEBUG_MINMAX 1

uniform vec4 viewport;
uniform mat4 modelViewProjection;

uniform int rowStride;

uniform float kernelWidth;
uniform bool attenuateSubpixel;

layout(std430, binding = 3) buffer ValueSSBO {
    float values[];
};

layout(std430, binding = 4) buffer FlagBuffer {
    uint flags[];
};

out float vsKernelSize;
out float vsPixelKernelSize;
out float vsValue;
out vec4 vsValueColor;
out vec4 vsPosition;

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

void main() {
    const Plot plot = plots[gl_InstanceID];
    const int rowOffset = gl_VertexID * rowStride;

    // Transform kernel size to screen space.
    const vec4 ndcKernelSize = modelViewProjection * vec4(kernelWidth, kernelWidth, 0.0, 0.0);
    const vec2 screenKernelSize = ndcKernelSize.xy * viewport.zw;
    vsKernelSize = max(screenKernelSize.x, screenKernelSize.y);

    if (attenuateSubpixel) {
        // Ensure a minimum pixel size to attenuate alpha depending on subpixels.
        vsPixelKernelSize = max(vsKernelSize, 1.0);
    } else {
        vsPixelKernelSize = vsKernelSize;
    }
    gl_PointSize = vsPixelKernelSize;

    vsValue = normalizeValue(values[rowOffset + valueColumn]);
    vsValueColor = flagifyColor(tflookup(vsValue), flags[gl_VertexID]);

    vsPosition = valuesToPosition(plot, 
        vec2(values[rowOffset + plot.indexX],
        values[rowOffset + plot.indexY]));
    if (bitflag_test(flags[gl_VertexID], FLAG_ENABLED | FLAG_FILTERED, FLAG_ENABLED)) {
        gl_Position = modelViewProjection * vsPosition;
		gl_ClipDistance[0] = 1.0;
    } else {
        gl_ClipDistance[0] = -1.0; // clipping cheat
    }
}

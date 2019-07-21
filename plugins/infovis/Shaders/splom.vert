//#define DEBUG_MINMAX 1

uniform vec4 viewport;
uniform mat4 modelViewProjection;

uniform sampler1D colorTable;
uniform int colorCount;
uniform int colorColumn;
uniform vec2 colorColumnMinMax;

uniform int rowStride;

uniform float kernelWidth;
uniform float alphaScaling;
uniform bool attenuateSubpixel;

layout(std430, binding = 3) buffer ValueSSBO {
    float values[];
};

out vec4 vsColor;
out vec4 vsPosition;
out float vsKernelSize;
out float vsPixelKernelSize;

void main(void) {
    const Plot plot = plots[gl_InstanceID];
    const int rowOffset = gl_VertexID * rowStride;

    // Fetch color from table.
    const float colorValue = values[rowOffset + colorColumn];
    const float colorValueNormalized = (colorValue - colorColumnMinMax.x) / 
        (colorColumnMinMax.y - colorColumnMinMax.x);
    const float colorOffset = colorValueNormalized + 0.5 / float(colorCount);
    vsColor = texture(colorTable, colorOffset);

    // Map value pair to position.
    const vec2 point = vec2(values[rowOffset + plot.indexX],
                            values[rowOffset + plot.indexY]);
    vec2 unitPoint = vec2((point.x - plot.minX) / (plot.maxX - plot.minX),
                                (point.y - plot.minY) / (plot.maxY - plot.minY));
#if DEBUG_MINMAX
    // To debug column min/max and thus normalization, we clamp the result and set it to red.
    vec2 unitPointClamped = clamp(unitPoint, vec2(0.0), vec2(1.0));
    if (unitPoint != unitPointClamped) {
        unitPoint = unitPointClamped;
        vsColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
#endif
    vsPosition = vec4(unitPoint.x * plot.sizeX + plot.offsetX,
                      unitPoint.y * plot.sizeY + plot.offsetY,
                      0.0, 1.0);
    gl_Position = modelViewProjection * vsPosition;

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
}

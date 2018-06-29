uniform vec4 viewport;
uniform mat4 modelViewProjection;

uniform sampler1D colorTable;
uniform uint colorCount;
uniform uint colorColumn;

uniform int rowStride;

uniform float kernelWidth;
uniform float alphaScaling;
uniform bool attenuateSubpixel;

layout(std430, binding = 3) buffer ValueSSBO {
    float values[];
};

out vec4 vsColor;
out float vsSize;
out float vsPixelSize;

void main(void) {
    const Plot plot = plots[gl_InstanceID];
    const int rowOffset = gl_VertexID * rowStride;

    // Fetch color from table.
    const float colorIndex = values[rowOffset + colorColumn];
    const float colorOffset = clamp(colorIndex / float(colorCount - 1), 0.0, 1.0) + 0.5 / float(colorCount);
    vsColor = texture(colorTable, colorOffset);

    // Map value pair to position.
    const vec2 point = vec2(values[rowOffset + plot.indexX],
                            values[rowOffset + plot.indexY]);
    const vec2 unitPoint = vec2((point.x - plot.minX) / (plot.maxX - plot.minX),
                                (point.y - plot.minY) / (plot.maxY - plot.minY));
    const vec4 position = vec4(unitPoint.x * plot.sizeX + plot.offsetX,
                      unitPoint.y * plot.sizeY + plot.offsetY,
                      0.0, 1.0);
    gl_Position = modelViewProjection * position;

     // Transform kernel size to screen space.
    const vec4 ndcSize = modelViewProjection * vec4(kernelWidth, kernelWidth, 0.0, 0.0);
    const vec2 screenSize = ndcSize.xy * viewport.zw;
    vsSize = max(screenSize.x, screenSize.y);

    if (attenuateSubpixel) {
        // Ensure a minimum pixel size to attenuate alpha depending on subpixels.
        vsPixelSize = max(vsSize, 1.0);
    } else {
        vsPixelSize = vsSize;
    }
    gl_PointSize = vsPixelSize;
}

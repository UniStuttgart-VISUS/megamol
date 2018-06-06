uniform vec4 viewport;
uniform mat4 modelViewProjection;

uniform sampler1D colorTable;
uniform uvec2 colorConsts;

uniform int rowStride;

uniform float kernelWidth;
uniform float alphaScaling;
uniform bool attenuateSubpixel;

layout(std430, binding = 3) buffer ValueSSBO {
    float values[];
};

out vec4 vsPosition;
out vec4 vsColor;
out float vsEffectiveDiameter;

void main(void) {
    const Plot plot = plots[gl_InstanceID];
    const int rowOffset = gl_VertexID / rowStride * rowStride;

    // Map value pair to position.
    const vec2 point = vec2(values[rowOffset + plot.indexX],
                            values[rowOffset + plot.indexY]);
    const vec2 unitPoint = vec2((point.x - plot.minX) / (plot.maxX - plot.minX),
                                (point.y - plot.minY) / (plot.maxY - plot.minY));
    vsPosition = vec4(unitPoint.x * plot.sizeX + plot.offsetX,
                      unitPoint.y * plot.sizeY + plot.offsetY,
                      0.0f, 1.0f);

    const float colorCount = float(colorConsts[1]);
    if (colorCount != 0) {
        // Fetch color from table.
        const float colorIndex = values[rowOffset + colorConsts[0]];
        const float colorOffset = clamp(colorIndex, 0.0, 1.0) * (1.0 - 1.0 / colorCount)
                                  + 0.5 / colorCount;
        vsColor = texture(colorTable, colorOffset);
    } else {
        // Use fallback color.
        vsColor = vec4(1.0f);
    }

    if (attenuateSubpixel) {
        // Attenuate alpha depending on subpixels.
        vsEffectiveDiameter = kernelWidth;
    } else {
        vsEffectiveDiameter = 1.0;
    }

    gl_Position = modelViewProjection * vsPosition;
    gl_PointSize = kernelWidth;
}

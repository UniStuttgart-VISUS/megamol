struct Plot {
    uvec2 index;
    vec2 offset;
    vec2 size;
};

uniform vec4 viewport;
uniform mat4 modelViewProjection;

uniform float kernelWidth;
uniform float alphaScaling;
uniform bool attenuateSubpixel;

uniform sampler1D colorTable;
uniform uvec2 colorConsts;

layout(std430, packed, binding = 3) buffer PlotSSBO {
    Plot plots[];
};

layout(std430, packed, binding = 2) buffer RowSSBO {
    float values[];
};

out vec4 vsPosition;
out vec4 vsColor;
out float vsEffectiveDiameter;

void main(void) {
    const Plot plot = plots[gl_InstanceID];
    const vec2 position = vec2(values[gl_VertexID + plot.index.x],
                               values[gl_VertexID + plot.index.y]);

    // Map value to position.
    vsPosition = vec4(position.x / plot.size.x + plot.offset.x,
                      position.y / plot.size.y + plot.offset.y,
                      0.0f, 1.0f);

    const float colorCount = float(colorConsts.y);
    if (colorCount != 0) {
        // Fetch color from table.
        const float colorIndex = values[gl_VertexID + colorConsts.x];
        const float colorOffset = clamp(colorIndex, 0.0, 1.0) * (1.0 - 1.0 / colorCount)
                                  + 0.5 / colorCount;
        vsColor = texture(colorTable, colorOffset);
    } else {
        // Use fallback color.
        vsColor = vec4(1.0f);
    }

    if (attenuateSubpixel) {
        // Attenuate alpha depending on subpixels.
        vsEffectiveDiameter = gl_PointSize;
    } else {
        vsEffectiveDiameter = 1.0;
    }

    gl_Position = modelViewProjection * vsPosition;
    gl_PointSize = kernelWidth;
}

struct Plot {
    uvec2 index;
    vec2 offset;
    vec2 size;
};

uniform mat4 modelViewProjection;

layout(std430, packed, binding = 2) buffer PlotSSBO {
    Plot plots[];
};

void main(void) {
    const uint plotIndex = gl_VertexID / 4;
    const Plot plot = plots[plotIndex];

    // Map plot to position.
    const uint vertexIndex = gl_VertexID % 4;
    vec2 position;
    switch (vertexIndex) {
    case 0:
    case 2:
        position = vec2(plot.offset.x, plot.offset.y);
        break;
    case 1:
        position = vec2(plot.offset.x, plot.offset.y + plot.size.y);
        break;
    case 3:
        position = vec2(plot.offset.x + plot.size.x, plot.offset.y);
        break;
    }

    gl_Position = modelViewProjection * vec4(position, 0.0f, 1.0f);
}

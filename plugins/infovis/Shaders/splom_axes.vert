struct Plot {
    uvec2 index;
    vec2 offset;
    vec2 size;
};

uniform mat4 modelViewProjection;
uniform vec4 axisColor;

out vec4 vsColor;

layout(std430, packed, binding = 2) buffer PlotSSBO {
    Plot plots[];
};

void main(void) {
    const Plot plot = plots[gl_InstanceID];

    // Map plot to axis vertices.
    const uint vertexIndex = gl_VertexID;
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

    vsColor = axisColor;
    
    gl_Position = modelViewProjection * vec4(position, 0.0f, 1.0f);
}

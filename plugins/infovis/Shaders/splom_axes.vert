uniform mat4 modelViewProjection;
uniform vec4 axisColor;

out vec4 vsColor;

void main(void) {
    const Plot plot = plots[gl_InstanceID];

    // Map plot to axis vertices.
    const uint vertexIndex = gl_VertexID;
    vec2 position;
    switch (vertexIndex) {
    case 0:
    case 2:
        position = vec2(plot.offsetX, plot.offsetY);
        break;
    case 1:
        position = vec2(plot.offsetX, plot.offsetY + plot.sizeY);
        break;
    case 3:
        position = vec2(plot.offsetX + plot.sizeX, plot.offsetY);
        break;
    }

    vsColor = axisColor;
    
    gl_Position = modelViewProjection * vec4(position, 0.0f, 1.0f);
}

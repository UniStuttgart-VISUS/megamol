uniform mat4 modelViewProjection;
uniform vec4 axisColor;
uniform uint numTicks;
uniform float tickSize;
uniform bool skipInnerTicks;

out vec4 vsColor;

const int CORNER_TL = 0;
const int CORNER_BL = 1;
const int CORNER_BR = 2;
const int CORNER_TR = 3;

// Maps a plot and a vertex index to world space.
vec2 corner(const Plot plot, const uint vertexIndex) {
    switch (vertexIndex) {
    case CORNER_TL:
        // Top-left.
        return vec2(plot.offsetX, plot.offsetY + plot.sizeY);
    case CORNER_BL:
        // Bottom-left.
        return vec2(plot.offsetX, plot.offsetY);
    case CORNER_BR:
        // Bottom-right.
        return vec2(plot.offsetX + plot.sizeX, plot.offsetY);
    case CORNER_TR:
        // Top-right.
        return vec2(plot.offsetX + plot.sizeX, plot.offsetY + plot.sizeY);
    }
}

vec2 tick(const Plot plot, const uint vertexIndex) {
    const uint verticesPerAxis = numTicks * 2;
    const uint axisIndex = vertexIndex / verticesPerAxis;
    const uint tickIndex = vertexIndex % verticesPerAxis;
    const bool isHorizontal = (axisIndex == 0);
    const bool isNextToDiagnoal = (plot.indexX - plot.indexY) == -1;
    vec2 offset = vec2(0);
    if (tickIndex % 2 == 0) {
        offset = vec2(
            isHorizontal ? 0 : tickSize,
            isHorizontal ? -tickSize : 0
        );
    }
    if (!skipInnerTicks || isNextToDiagnoal) {
        return offset + mix(
            corner(plot, isHorizontal ? CORNER_BL : CORNER_BR), 
            corner(plot, isHorizontal ? CORNER_BR : CORNER_TR),
            float(tickIndex / 2) / (numTicks - 1)
        );
    } else {
        return vec2(0);
    }
}

void main(void) {
    const Plot plot = plots[gl_InstanceID];

    // Map index to border and tick positions.
    vec2 position = vec2(0,0);
    switch (gl_VertexID) {
    case 0:
    case 7:
        position = corner(plot, CORNER_BR);
        break;
    case 1:
    case 2:
        position = corner(plot, CORNER_BL);
        break;
    case 3:
    case 4:
        position = corner(plot, CORNER_TL);
        break;
    case 5:
    case 6:
        position = corner(plot, CORNER_TR);
        break;
    default:
        position = tick(plot, uint(gl_VertexID - 8));
        break;
    }

    vsColor = axisColor;
    
    gl_Position = modelViewProjection * vec4(position, 0.0f, 1.0f);
}

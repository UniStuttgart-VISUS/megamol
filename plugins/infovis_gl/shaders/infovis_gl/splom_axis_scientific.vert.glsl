#version 430

#include "splom_common/splom_plots.inc.glsl"

uniform mat4 modelViewProjection;
uniform vec4 axisColor;


out vec4 vsColor;
out vec2 vsFragCoord; // in objectSpace
out vec2 vsSmallSteppSize;

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

void main(void) {
    const Plot plot = plots[gl_InstanceID];

    // Map index to border and tick positions.
    vec2 position = vec2(0, 0);
    switch (gl_VertexID) {
    case 0:
        position = corner(plot, CORNER_BL);
        vsFragCoord = vec2(plot.minX, plot.minY);
        break;
    case 1:
        position = corner(plot, CORNER_BR);
        vsFragCoord = vec2(plot.maxX, plot.minY);
        break;
    case 2:
        position = corner(plot, CORNER_TR);
        vsFragCoord = vec2(plot.maxX, plot.maxY);
        break;
    case 3:
        position = corner(plot, CORNER_TL);
        vsFragCoord = vec2(plot.minX, plot.maxY);
        break;
    default:
        position = vec2(0, 0); // failed
        break;
    }

    vsColor = axisColor;
    vsSmallSteppSize = vec2(plot.smallTickX, plot.smallTickY);

    gl_Position = modelViewProjection * vec4(position, 0.0f, 1.0f);
}

#version 430

#include "splom_common/splom_plots.inc.glsl"

uniform mat4 modelViewProjection;
uniform vec4 axisColor;
uniform uint numTicks;
uniform float tickLength;
uniform bool redundantTicks;
uniform bool drawOuter;
uniform bool drawDiagonal;
uniform bool invertY;
uniform int columnCount;

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
    return vec2(0.0f);
}

vec2 tick(const Plot plot, const uint vertexIndex) {
    const uint verticesPerAxis = numTicks * 2;
    const uint axisIndex = vertexIndex / verticesPerAxis; // 0 = bottom, 1 = rightm 2 = top, 3 = left
    const uint tickIndex = vertexIndex % verticesPerAxis;

    const bool isNextToOuterX = plot.indexX == 0;
    const bool isNextToOuterY = plot.indexY == columnCount - 1;
    const bool isNextToDiagnoal = ((plot.indexX - plot.indexY) == -1);

    const bool drawLeft = drawOuter && (redundantTicks || isNextToOuterX);
    const bool drawRight = drawDiagonal && (redundantTicks || isNextToDiagnoal);
    const bool drawBottom = invertY ? (drawOuter && (redundantTicks || isNextToOuterY)) : (drawDiagonal && (redundantTicks || isNextToDiagnoal));
    const bool drawTop = invertY ? (drawDiagonal && (redundantTicks || isNextToDiagnoal)) : (drawOuter && (redundantTicks || isNextToOuterY));

    if ((axisIndex == 0 && !drawBottom) || (axisIndex == 1 && !drawRight) || (axisIndex == 2 && !drawTop) || (axisIndex == 3 && !drawLeft)) {
        return vec2(0);
    }

    float t = float(tickIndex / 2) / float(numTicks - 1);
    vec2 pos = vec2(0.0);
    if (axisIndex == 0) {
        pos = mix(corner(plot, CORNER_BL), corner(plot, CORNER_BR), t);
        if (tickIndex % 2 == 1) {
            pos.y -= tickLength;
        }
    } else if (axisIndex == 1) {
        pos = mix(corner(plot, CORNER_BR), corner(plot, CORNER_TR), t);
        if (tickIndex % 2 == 1) {
            pos.x += tickLength;
        }
    } else if (axisIndex == 2) {
        pos = mix(corner(plot, CORNER_TL), corner(plot, CORNER_TR), t);
        if (tickIndex % 2 == 1) {
            pos.y += tickLength;
        }
    } else if (axisIndex == 3) {
        pos = mix(corner(plot, CORNER_BL), corner(plot, CORNER_TL), t);
        if (tickIndex % 2 == 1) {
            pos.x -= tickLength;
        }
    }
    return pos;
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

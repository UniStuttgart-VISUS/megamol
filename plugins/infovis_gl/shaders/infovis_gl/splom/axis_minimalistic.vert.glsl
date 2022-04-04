#version 450

#include "common/plots.inc.glsl"
#include "../common/quad_vertices.inc.glsl"

uniform mat4 modelViewProjection;
uniform uint numTicks;
uniform float tickLength;
uniform bool redundantTicks;
uniform bool drawOuter;
uniform bool drawDiagonal;
uniform bool invertY;
uniform int columnCount;
uniform float axisWidth;
uniform ivec2 viewSize;

void main() {
    // Instance corresponds to unique line, 4 + 4 * numTicks belong to each Plot
    const uint plotIdx = gl_InstanceID / (4 + 4 * numTicks);
    const uint lineIdx = gl_InstanceID % (4 + 4 * numTicks);

    const Plot plot = plots[plotIdx];

    vec2 vertexPos = quadVertexPosition();
    vec2 pos = vec2(plot.offsetX, plot.offsetY);

    bool isHorizontal = false;
    bool cornerCorrection = false;

    if (lineIdx < 4) {
        // Draw a box around each plot with 4 lines.
        switch (lineIdx) {
            case 0: // bottom
                pos.x += vertexPos.x * plot.sizeX;
                isHorizontal = true;
                break;
            case 1: // top
                pos.x += vertexPos.x * plot.sizeX;
                pos.y += plot.sizeY;
                isHorizontal = true;
                break;
            case 2: // left
                pos.y += vertexPos.y * plot.sizeY;
                break;
            case 3: // right
                pos.y += vertexPos.y * plot.sizeY;
                pos.x += plot.sizeX;
                break;
        }
        cornerCorrection = true;
    } else {
        // Draw ticks.
        const uint tickIdx = lineIdx - 4;
        const uint axisIndex = tickIdx / numTicks;

        const bool isNextToOuterX = plot.indexX == 0;
        const bool isNextToOuterY = plot.indexY == columnCount - 1;
        const bool isNextToDiagnoal = ((plot.indexX - plot.indexY) == -1);

        const bool drawLeft = drawOuter && (redundantTicks || isNextToOuterX);
        const bool drawRight = drawDiagonal && (redundantTicks || isNextToDiagnoal);
        const bool drawBottom = invertY ? (drawOuter && (redundantTicks || isNextToOuterY)) : (drawDiagonal && (redundantTicks || isNextToDiagnoal));
        const bool drawTop = invertY ? (drawDiagonal && (redundantTicks || isNextToDiagnoal)) : (drawOuter && (redundantTicks || isNextToOuterY));

        if ((axisIndex == 0 && !drawBottom) || (axisIndex == 1 && !drawTop) || (axisIndex == 2 && !drawLeft) || (axisIndex == 3 && !drawRight)) {
            pos = vec2(0.0f);
        } else {
            const float t = float(tickIdx % numTicks) / float(numTicks - 1);
            switch (axisIndex) {
                case 0: // bottom
                    pos.x += t * plot.sizeX;
                    pos.y += -tickLength + vertexPos.y * tickLength;
                    break;
                case 1: // top
                    pos.x += t * plot.sizeX;
                    pos.y += plot.sizeY + vertexPos.y * tickLength;
                    break;
                case 2: // left
                    pos.y += t * plot.sizeY;
                    pos.x += -tickLength + vertexPos.x * tickLength;
                    isHorizontal = true;
                    break;
                case 3: // right
                    pos.y += t * plot.sizeY;
                    pos.x += plot.sizeX + vertexPos.x * tickLength;
                    isHorizontal = true;
                    break;
            }
        }
    }

    pos = (modelViewProjection * vec4(pos, 0.0f, 1.0f)).xy;

    // Apply line width
    if (isHorizontal || cornerCorrection) {
        pos.y += (vertexPos.y - 0.5f) * axisWidth * 2.0f / viewSize.y;
    }
    if (!isHorizontal || cornerCorrection) {
        pos.x += (vertexPos.x - 0.5f) * axisWidth * 2.0f / viewSize.x;
    }

    gl_Position = vec4(pos, 0.0f, 1.0f);
}

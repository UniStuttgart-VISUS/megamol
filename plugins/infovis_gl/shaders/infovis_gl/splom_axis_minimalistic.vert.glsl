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
uniform float axisWidth;
uniform ivec2 viewSize;

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

vec2 tick(const Plot plot, const uint tickID) {
    const uint axisIndex = tickID / numTicks;

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

    // relative position on each edge for current tick
    float t = float(tickID % (numTicks)) / float(numTicks - 1);
    vec2 pos = vec2(0.0);
    float width = 0.5 * (modelViewProjection * vec4(viewSize, 0.0, 1.0)).x;
    float height = 0.5 * (modelViewProjection * vec4(viewSize, 0.0, 1.0)).y;
    const float tickLengthConstant = tickLength * (corner(plot, CORNER_TR).y - corner(plot, CORNER_BR).y) / 20.0;
    // each instance only contains 4 vertices
    if (axisIndex == 0) {
        pos = mix(corner(plot, CORNER_BL), corner(plot, CORNER_BR), t);
        pos.y -= float(gl_VertexID / 2) * tickLengthConstant;
        pos.x -= (float((gl_VertexID % 2)) - 0.5) * axisWidth / width;
    } else if (axisIndex == 1) {
        pos = mix(corner(plot, CORNER_BR), corner(plot, CORNER_TR), t);
        pos.x += float(gl_VertexID / 2) * tickLengthConstant;
        pos.y -= (float((gl_VertexID % 2)) - 0.5) * axisWidth / height;
    } else if (axisIndex == 2) {
        pos = mix(corner(plot, CORNER_TL), corner(plot, CORNER_TR), t);
        pos.y += float(gl_VertexID / 2) * tickLengthConstant;
        pos.x -= (float((gl_VertexID % 2)) - 0.5) * axisWidth / width;
    } else if (axisIndex == 3) {
        pos = mix(corner(plot, CORNER_BL), corner(plot, CORNER_TL), t);
        pos.x -= float(gl_VertexID / 2) * tickLengthConstant;
        pos.y -= (float((gl_VertexID % 2)) - 0.5) * axisWidth / height;
    }
    return pos;
}

void main(void) {
    // Instance corresponds to unique line
    // 4 + 4 * numTicks belong to each Plot
    const Plot plot = plots[gl_InstanceID / (4 + 4 * numTicks)];

    // quick and dirty conversion of pixel dependant size to witch space
    float width = 0.5 * (modelViewProjection * vec4(viewSize, 0.0, 1.0)).x;
    float height = 0.5 * (modelViewProjection * vec4(viewSize, 0.0, 1.0)).y;

    // Map index to border and tick positions.
    vec2 position = vec2(0,0);
    switch (gl_InstanceID % (4 + 4 * numTicks)) {
    case 0: // bottom line
        // true for the first two vertices of each line => left side
        if( gl_VertexID / 2 == 0){
            position = corner(plot, CORNER_BL);
        } else { //right side of bottom line
            position = corner(plot, CORNER_BR);
        }
        // offset to generate line width
        position.y += (float(gl_VertexID % 2) - 0.5) * axisWidth / height;
        // second offset to get 45° angle on corners
        if (gl_VertexID % 2 == 1 ^^ (gl_VertexID / 2) % 2 == 1) {
            position.x += 0.5 * axisWidth / width;
        } else {
            position.x -= 0.5 * axisWidth / width;
        }
        break;
    case 1: // right
        if (gl_VertexID / 2 == 0) {
            position = corner(plot, CORNER_BR);
        } else {
            position = corner(plot, CORNER_TR);
        }
        position.x += (float(gl_VertexID % 2) - 0.5) * axisWidth / width;
        if(gl_VertexID % 2 == 1 ^^ (gl_VertexID / 2) % 2 == 1) {
            position.y -= 0.5 * axisWidth / height;
        } else {
            position.y += 0.5 * axisWidth / height;
        }
        break;
    case 2: // top
        if (gl_VertexID / 2 == 0) {
            position = corner(plot, CORNER_TR);
        } else {
            position = corner(plot, CORNER_TL);
        }
        position.y += (float(gl_VertexID % 2) - 0.5) * axisWidth / height;
        if(gl_VertexID % 2 == 1 ^^ (gl_VertexID / 2) % 2 == 1) {
            position.x += 0.5 * axisWidth / width;
        } else {
            position.x -= 0.5 * axisWidth / width;
        }
        break;
    case 3: // left
        if (gl_VertexID / 2 == 0) {
            position = corner(plot, CORNER_TL);
        } else {
            position = corner(plot, CORNER_BL);
        }
        position.x -= (float(gl_VertexID % 2) - 0.5) * axisWidth / width;
        if(gl_VertexID % 2 == 1 ^^ (gl_VertexID / 2) % 2 == 1) {
            position.y += 0.5 * axisWidth / height;
        } else {
            position.y -= 0.5 * axisWidth / height;
        }
        break;
    default: // catch for all ticks
        position = tick(plot, gl_InstanceID % (4 + 4 * numTicks) - 4);
        break;
    }

    vsColor = axisColor;

    gl_Position = modelViewProjection * vec4(position, 0.0f, 1.0f);
}

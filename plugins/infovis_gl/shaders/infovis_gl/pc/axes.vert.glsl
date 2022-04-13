#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/quad_vertices.inc.glsl"

uniform float lineWidth = 1.0f;
uniform ivec2 viewSize = ivec2(1, 1);
uniform uint numTicks = 3;
uniform float tickLength = 4.0f;
uniform int pickedAxis = -1;
uniform ivec2 pickedFilter = ivec2(-1, -1);
uniform vec4 axesColor = vec4(1.0f);
uniform vec4 filterColor = vec4(1.0f);

out vec4 color;

void main() {
    const uint axisIdx = gl_InstanceID % dimensionCount;
    const uint lineIdx = gl_InstanceID / dimensionCount;

    const uint dimensionIdx = axisIndirection[axisIdx];
    const float lineWidthHalf = lineWidth / 2.0f;
    vec2 vertexPos = quadVertexPosition();

    if (dimensionIdx == pickedAxis) {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        color = axesColor;
    }

    if (lineIdx == 0) {
        // Axis
        vec4 pos = projMx * viewMx * pc_axisVertex(axisIdx, vertexPos.y);
        pos.x += (-lineWidthHalf + vertexPos.x * lineWidth) * 2.0f / viewSize.x;
        gl_Position = pos;
    } else if (lineIdx >= 1 && lineIdx <= numTicks) {
        // Axis Ticks
        const uint tickIdx = lineIdx - 1;
        vec4 pos = pc_axisVertex(axisIdx, float(tickIdx) / float(numTicks - 1));
        pos.x += -tickLength / 2.0f + vertexPos.x * tickLength;
        pos = projMx * viewMx * pos;
        pos.y += (-lineWidthHalf + vertexPos.y * lineWidth) * 2.0f / viewSize.y;
        gl_Position = pos;
    } else {
        // Filter Indicators
        const uint indicatorIdx = lineIdx - 1 - numTicks;
        const bool isTop = (indicatorIdx / 2) == 1;
        const bool isLeft = (indicatorIdx % 2) == 0;

        if (!isTop) {
            vertexPos.y -= 1.0f;
        }
        if (!isLeft) {
            vertexPos.x -= 1.0f;
        }

        float filterValue = isTop ? filters[dimensionIdx].max : filters[dimensionIdx].min;
        vec4 pos = pc_axisVertex(axisIdx, pc_normalizeDimension(filterValue, dimensionIdx));

        pos.x += vertexPos.x * tickLength / 2.0f;
        pos.y += (isTop ? 1.0f : -1.0f) * abs(vertexPos.x) * tickLength / 2.0f;
        pos = projMx * viewMx * pos;
        pos.y += vertexPos.y * 4.0f * lineWidth * 2.0f / viewSize.y;
        gl_Position = pos;

        if (dimensionIdx == pickedFilter.x && (isTop ? 1 : 0) == pickedFilter.y) {
            color = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            color = filterColor;
        }
    }
}

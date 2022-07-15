#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"
#include "mmstd_gl/flags/bitflags.inc.glsl"

uniform bool useTransferFunction = true;
uniform vec4 itemColor = vec4(1.0f);
uniform int colorDimensionIdx = -1;
uniform uint itemTestMask = 0;
uniform uint itemPassMask = 0;
uniform bool useLineWidthInPixels = true;
uniform float lineWidth = 1.0f;
uniform ivec2 viewSize = ivec2(1, 1);

out vec4 color;

void main() {
    const uint itemIdx = gl_InstanceID;

    if (bitflag_test(flags[itemIdx], itemTestMask, itemPassMask)) {
        gl_ClipDistance[0] = 1.0;
    } else {
        gl_ClipDistance[0] = -1.0;
        gl_Position = vec4(0.0f);
        return;
    }

#ifndef TRIANGLES
    // Line strip
    const uint axisIdx = gl_VertexID;
    const uint dimensionIdx = axisIndirection[axisIdx];

    vec4 pos = pc_itemVertex(itemIdx, axisIdx);
    gl_Position = projMx * viewMx * pos;
#else
    // Triangle strip
    const uint axisIdx = gl_VertexID / 2;
    const bool isTop = gl_VertexID % 2 == 0;
    const uint dimensionIdx = axisIndirection[axisIdx];

    const float aspect = float(viewSize.x) / float(viewSize.y);

    vec4 vertex = projMx * viewMx * pc_itemVertex(itemIdx, axisIdx);
    vertex.y = vertex.y / aspect; // Map to uniform coordinate system.

    // Find direction vectors pointing to left and right neighbor axis.
    vec2 left = vec2(-1.0f, 0.0f);
    vec2 right = vec2(1.0f, 0.0f);
    if (axisIdx > 0) {
        vec4 vertexLeft = projMx * viewMx * pc_itemVertex(itemIdx, axisIdx - 1);
        vertexLeft.y = vertexLeft.y / aspect;
        left = normalize(vertexLeft.xy - vertex.xy);
    }
    if (axisIdx < dimensionCount - 1) {
        vec4 vertexRight = projMx * viewMx * pc_itemVertex(itemIdx, axisIdx + 1);
        vertexRight.y = vertexRight.y / aspect;
        right = normalize(vertexRight.xy - vertex.xy);
    }

    // Orthogonal directions (point to top of PCP).
    vec2 orthoLeft = vec2(left.y, -left.x);
    vec2 orthoRight = vec2(-right.y, right.x);

    // Half vector
    vec2 h = vec2(0.0f);
    if (axisIdx == 0) {
        h = vec2(0.0f, 1.0f);
        h /= dot(orthoRight, h);
    } else if (axisIdx >= dimensionCount - 1) {
        h = vec2(0.0f, 1.0f);
        h /= dot(orthoLeft, h);
    } else {
        h = normalize(orthoLeft + orthoRight);
        h /= dot(orthoRight, h);
    }

    // Set length to lineWidth / 2, measuerd in orthogonal direction.
    if (useLineWidthInPixels) {
        h *= lineWidth / float(viewSize.x);
    } else {
        const vec4 oneVec = projMx * viewMx * vec4(1.0f, 0.0f, 0.0f, 0.0f);
        h *= oneVec.x * lineWidth / 2.0f;
    }

    if (isTop) {
        vertex.xy += h;
    } else {
        vertex.xy -= h;
    }

    vertex.y *= aspect; // Map back to screen space.
    gl_Position = vertex;
#endif

    if (useTransferFunction) {
        float value = 0.0f;
        if (colorDimensionIdx >= 0) {
            value = pc_dataValue(itemIdx, colorDimensionIdx);
        } else {
            value = float(itemIdx) / float(itemCount - 1);
        }
        color = tflookup(value);
    } else {
        color = itemColor;
    }
}

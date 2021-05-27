#version 430

uniform mat4 modelViewProjection;

uniform float kernelWidth;

in float vsValue[];
in vec4 vsValueColor[];
in vec4 vsPosition[];
in int vsFiltered[];

out float gsValue;
out vec4 gsValueColor;
out vec2 gsLineSize;
out vec2 gsLineCoord;

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

void main(void) {
    // Filter
    if (vsFiltered[0] > 0 || vsFiltered[1] > 0) {
        return;
    }

    const vec2 p0 = vsPosition[0].xy;
    const vec2 p1 = vsPosition[1].xy;

    const vec2 lineTangent = normalize(p1 - p0);
    const vec2 lineNormal = vec2(-lineTangent.y, lineTangent.x);
    const float lineLength = length(p1 - p0);

    //XXX: keep this, as it might become a vertex attribute.
    const float ndcKernelWidth0 = kernelWidth;
    const float ndcKernelWidth1 = kernelWidth;

    // Construct triangle vertices.
    vec2 t0 = (p0 - lineTangent * ndcKernelWidth0) + lineNormal * ndcKernelWidth0;
    vec2 t1 = (p0 - lineTangent * ndcKernelWidth0) - lineNormal * ndcKernelWidth0;
    vec2 t2 = (p1 + lineTangent * ndcKernelWidth1) + lineNormal * ndcKernelWidth1;
    vec2 t3 = (p1 + lineTangent * ndcKernelWidth1) - lineNormal * ndcKernelWidth1;

    gsValue = vsValue[0];
    gsValueColor = vsValueColor[0];
    gsLineSize = vec2(ndcKernelWidth0, lineLength);
    gsLineCoord = vec2(-ndcKernelWidth0, ndcKernelWidth0);
    gl_Position = modelViewProjection * vec4(t0, 0.0, 1.0);
    EmitVertex();

    gsValue = vsValue[1];
    gsValueColor = vsValueColor[1];
    gsLineSize = vec2(ndcKernelWidth1, lineLength);
    gsLineCoord = vec2(lineLength  + ndcKernelWidth1, ndcKernelWidth1);
    gl_Position = modelViewProjection * vec4(t2, 0.0, 1.0);
    EmitVertex();

    gsValue = vsValue[0];
    gsValueColor = vsValueColor[0];
    gsLineSize = vec2(ndcKernelWidth0, lineLength);
    gsLineCoord = vec2(-ndcKernelWidth0, -ndcKernelWidth0);
    gl_Position = modelViewProjection * vec4(t1, 0.0, 1.0);
    EmitVertex();

    gsValue = vsValue[1];
    gsValueColor = vsValueColor[1];
    gsLineSize = vec2(ndcKernelWidth1, lineLength);
    gsLineCoord = vec2(lineLength + ndcKernelWidth1, -ndcKernelWidth1);
    gl_Position = modelViewProjection * vec4(t3, 0.0, 1.0);
    EmitVertex();
}

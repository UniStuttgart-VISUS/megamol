uniform vec4 viewport;
uniform float kernelWidth;

in vec4 vsColor[];
in float vsPixelKernelSize[];

out vec4 gsColor;
out vec2 gsLineCoord;
out vec2 gsLineSize;

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

void main(void) {
    const vec2 p0 = gl_in[0].gl_Position.xy;
    const vec2 p1 = gl_in[1].gl_Position.xy;

    //XXX: doing tangents/normals in NDC is dangerous for non-2D transformations.
    const vec2 lineTangent = normalize(p1 - p0);
    const vec2 lineNormal = vec2(-lineTangent.y, lineTangent.x);
    const float lineLength = length(p1 - p0);

    const float ndcKernelSize0 = vsPixelKernelSize[0] / max(viewport.z, viewport.w);
    const float ndcKernelSize1 = vsPixelKernelSize[1] / max(viewport.z, viewport.w);

    // Construct triangle vertices.
    vec2 t0 = (p0 - lineTangent * ndcKernelSize0) + lineNormal * ndcKernelSize0;
    vec2 t1 = (p0 - lineTangent * ndcKernelSize0) - lineNormal * ndcKernelSize0;
    vec2 t2 = (p1 + lineTangent * ndcKernelSize1) + lineNormal * ndcKernelSize1;
    vec2 t3 = (p1 + lineTangent * ndcKernelSize1) - lineNormal * ndcKernelSize1;

    gsColor = vsColor[0];
    gsLineCoord = vec2(-ndcKernelSize0, ndcKernelSize0);
    gsLineSize = vec2(ndcKernelSize0, lineLength);
    gl_Position = vec4(t0, 0.0, 1.0);
    EmitVertex();

    gsColor = vsColor[1];
    gsLineCoord = vec2(lineLength  + ndcKernelSize1, ndcKernelSize1);
    gsLineSize = vec2(ndcKernelSize1, lineLength);
    gl_Position = vec4(t2, 0.0, 1.0);
    EmitVertex();

    gsColor = vsColor[0];
    gsLineCoord = vec2(-ndcKernelSize0, -ndcKernelSize0);
    gsLineSize = vec2(ndcKernelSize0, lineLength);
    gl_Position = vec4(t1, 0.0, 1.0);
    EmitVertex();

    gsColor = vsColor[1];
    gsLineCoord = vec2(lineLength + ndcKernelSize1, -ndcKernelSize1);
    gsLineSize = vec2(ndcKernelSize1, lineLength);
    gl_Position = vec4(t3, 0.0, 1.0);
    EmitVertex();
}

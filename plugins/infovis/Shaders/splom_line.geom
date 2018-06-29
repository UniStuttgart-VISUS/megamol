uniform vec4 viewport;
uniform float kernelWidth;

in vec4 vsColor[];

out vec4 gsColor;
out vec2 gsLineCoord;
//out vec2 gsLineWidth;

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

void main(void) {
    const vec2 p0 = gl_in[0].gl_Position.xy;
    const vec2 p1 = gl_in[1].gl_Position.xy;

    const vec2 lineTangent = normalize(p1 - p0);
    const vec2 lineNormal = vec2(-lineTangent.y, lineTangent.x);
    const float lineLength = length(p1 - p0);
    //gsLineWidth = vec2(kernelWidth, lineLength);

    // Construct triangle vertices.
    vec2 t0 = (p0 - lineTangent * kernelWidth) + lineNormal * kernelWidth;
    vec2 t1 = (p0 - lineTangent * kernelWidth) - lineNormal * kernelWidth;
    vec2 t2 = (p1 + lineTangent * kernelWidth) + lineNormal * kernelWidth;
    vec2 t3 = (p1 + lineTangent * kernelWidth) - lineNormal * kernelWidth;

    gsColor = vsColor[0];
    gsLineCoord = (p0 - lineTangent * kernelWidth);
    gl_Position = vec4(t0, 0.0, 1.0);
    EmitVertex();

    gsColor = vsColor[1];
    gsLineCoord = (p1 + lineTangent * kernelWidth);
    gl_Position = vec4(t2, 0.0, 1.0);
    EmitVertex();

    gsColor = vsColor[0];
    gsLineCoord = (p0 - lineTangent * kernelWidth);
    gl_Position = vec4(t1, 0.0, 1.0);
    EmitVertex();

    gsColor = vsColor[1];
    gsLineCoord = (p1 + lineTangent * kernelWidth);
    gl_Position = vec4(t3, 0.0, 1.0);
    EmitVertex();
}

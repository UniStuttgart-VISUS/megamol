#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/quad_vertices.inc.glsl"

uniform vec2 strokeStart = vec2(0.0f, 0.0f);
uniform vec2 strokeEnd = vec2(0.0f, 0.0f);
uniform ivec2 viewSize = ivec2(1, 1);
uniform float lineWidth = 1.0f;

void main() {
    const float aspect = float(viewSize.x) / float(viewSize.y);
    const vec2 pos = quadVertexPosition();

    vec2 from = (projMx * viewMx * vec4(strokeStart, 0.0f, 1.0f)).xy;
    from.y /= aspect; // Map to uniform coordinate system.
    vec2 to = (projMx * viewMx * vec4(strokeEnd, 0.0f, 1.0f)).xy;
    to.y /= aspect;

    vec2 dir = to - from;
    vec2 ortho = normalize(vec2(-dir.y, dir.x));
    ortho *= lineWidth / float(viewSize.x); // Set length to lineWidth / 2.

    vec2 vertex = from + pos.x * dir + (pos.y * 2.0f - 1.0f) * ortho;
    vertex.y *= aspect;

    gl_Position = vec4(vertex, 0.0f, 1.0f);
}

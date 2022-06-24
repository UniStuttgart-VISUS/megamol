/**
 * Returns position of a vertex based on gl_VertexID assuming the following draw call:
 * glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
 *
 * gl_VertexID:
 *   2---3
 *   | \ |
 *   0---1
 *
 * Result:
 *   (0,1)---(1,1)
 *     |   \   |
 *   (0,0)---(1,0)
 */
vec2 quadVertexPosition() {
    return vec2(float(gl_VertexID % 2), float((gl_VertexID % 4) / 2));
}

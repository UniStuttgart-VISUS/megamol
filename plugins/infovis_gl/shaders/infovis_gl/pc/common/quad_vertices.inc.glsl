/**
 * Returns position of a vertex based on gl_VertexID assuming the following draw call:
 * glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
 *   2---3
 *   | \ |
 *   0---1
 */
vec2 quadVertexPosition() {
    if (gl_VertexID == 0) {
        return vec2(0.0f, 0.0f);
    } else if (gl_VertexID == 1) {
        return vec2(1.0f, 0.0f);
    } else if (gl_VertexID == 2) {
        return vec2(0.0f, 1.0f);
    } else if (gl_VertexID == 3) {
        return vec2(1.0f, 1.0f);
    }
    return vec2(0.0f, 0.0f);
}

// *******************************************************
// Transforms 2D ??? position to screen space coordinates.
//
// *******************************************************
vec2 toScreenSpace(const in vec2 position, const in vec2 viewport) {
    return (((position + vec2(1.0, 1.0)) / vec2(2.0, 2.0)) * viewport);
}

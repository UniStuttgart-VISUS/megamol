in vec4 gsColor;
in vec2 gsLineCoord;
in vec2 gsLineSize;

out vec4 fsColor;

// TODO: moving Gaussian please
float movingCircle(vec2 position, vec2 radius) {
    // Intersection between circle and line.
    float delta = pow(radius.x, 2.0) - pow(position.y, 2.0);
    if (delta <= 0) {
        // Less than two intersections.
        return 0;
    }
    delta = sqrt(delta);

    float x1 = position.x + delta;
    float x2 = position.x - delta;
    if (x1 < 0) {
        x1 = 0;
    } else if (x1 > radius.y) {
        x1 = radius.y;
    }
    if (x2 < 0) {
        x2 = 0;
    } else if (x2 > radius.y) {
        x2 = radius.y;
    }

    return abs(x1 - x2) / (2.0 * radius.x);
}

void main(void) {
    const float alpha = movingCircle(gsLineCoord, gsLineSize);
    fsColor = vec4(gsColor.rgb, gsColor.a * alpha);
}

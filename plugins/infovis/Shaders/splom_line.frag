uniform float alphaScaling;

in vec4 gsColor;
in vec2 gsLineCoord;
in vec2 gsLineSize;

out vec4 fsColor;

// TODO: moving Gaussian please
float movingCircle(vec2 position, float width, float len) {
    // Intersection between circle and center line.
    float delta = pow(width, 2.0) - pow(position.y, 2.0);
    if (delta <= 0) {
        // Less than two intersections.
        return 0;
    }
    delta = sqrt(delta);

    // Compute two intersected center points.
    float x1 = position.x + delta;
    float x2 = position.x - delta;
    if (x1 < 0) {
        x1 = 0;
    } else if (x1 > len) {
        x1 = len;
    }
    if (x2 < 0) {
        x2 = 0;
    } else if (x2 > len) {
        x2 = len;
    }

    // Compute the distance between the two center points.
    return abs(x1 - x2) / (2.0 * width);
}

void main(void) {
    float alpha = movingCircle(gsLineCoord, gsLineSize.x, gsLineSize.y);
    alpha *= alphaScaling;
    fsColor = vec4(gsColor.rgb, gsColor.a * alpha);
}

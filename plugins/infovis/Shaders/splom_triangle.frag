uniform int contourLevels;
uniform vec4 contourColor;

in vec3 vsPoint;

layout(location = 0) out vec4 fsColor;

float triangle(float x) {
    return abs(fract(x - 0.5) - 0.5);
}

//TODO: replace this crap with proper SDF-based intersection of contour isolevels, i.e., an uniform array!
float contourLine() {
    // Discard if contours are disabled.
    if (contourLevels <= 1.0) {
        return 0.0;
    }

    // Scale normalized value to specified number of contour levels.
    float level = float(contourLevels) * vsPoint.z;

    // Compute anti-aliased contour line.
    float line = triangle(level);
    line /= fwidth(level);

    return 1.0 - min(line, 1.0);
}

void main() {
    vec4 valueColor = vec4(vec3(vsPoint.z), 1.0);

    float contour = contourLine();
    fsColor = mix(valueColor, contourColor, contour);
}

#extension GL_ARB_derivative_control : enable

uniform int contourLevels;
uniform float contourWidth;
uniform vec4 contourColor;

flat in vec3 gsFlatValues;
in vec3 gsBarycentricPosition;

layout(location = 0) out vec4 fsColor;

float valueAt(vec3 barycentricPoint) {
    vec3 p = clamp(barycentricPoint, vec3(0.0), vec3(1.0));
    return p.x * gsFlatValues.x + p.y * gsFlatValues.y + p.z * gsFlatValues.z;
}

float contourDelta(float valueA, float valueB) {
    float levelA = int(valueA * contourLevels + 0.5) / float(contourLevels);
    float levelB = int(valueB * contourLevels + 0.5) / float(contourLevels);
    return abs(levelA - levelB) > 0.00001 ? 1.0 : 0.0;
}

float marchContour(float value) {
    // Discard if contours are disabled.
    if (contourLevels <= 0) {
        return 0.0;
    }
    // Raymarch along barycentric coordinate gradient in fragment-space,
    // while looking for a value change that implies a contour line.
    vec3 dx = dFdxFine(gsBarycentricPosition);
    vec3 dy = dFdyFine(gsBarycentricPosition);
    float stepLast = contourWidth / 2.0 + 1.0;
    for (float step = 1.0; step < stepLast; step += 0.25) {
        float d1 = valueAt(gsBarycentricPosition + dx * step);
        float d2 = valueAt(gsBarycentricPosition + dy * step);
        float d3 = valueAt(gsBarycentricPosition + (dx + dy) * step);
        float d4 = valueAt(gsBarycentricPosition - dx * step);
        float d5 = valueAt(gsBarycentricPosition - dy * step);
        float d6 = valueAt(gsBarycentricPosition + (-dx - dy) * step);
        float d7 = valueAt(gsBarycentricPosition + (dx - dy) * step);
        float d8 = valueAt(gsBarycentricPosition + (-dx + dy) * step);
        float deltaSum = contourDelta(value, d1) + contourDelta(value, d2) +
            contourDelta(value, d3) + contourDelta(value, d4) +
            contourDelta(value, d5) + contourDelta(value, d6) + 
            contourDelta(value, d7) + contourDelta(value, d8);
        if (deltaSum > 0.0) {
            return deltaSum / 8.0;
        }
    }
    return 0.0;
}

void main() {
    float value = valueAt(gsBarycentricPosition);
    fsColor = vec4(value, value, value, 1.0);
    //fsColor = vec4(gsBarycentricPosition / 4.0, 1.0);

    float contourness = marchContour(value);
    fsColor = mix(fsColor, contourColor, contourness);
}

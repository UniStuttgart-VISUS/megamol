uniform vec4 contourColor;
uniform float contourSize;
uniform float contourIsoValues[10];
uniform int contourIsoValueCount;

uniform sampler2D screenTexture;

in vec2 vsUV;

layout(location = 0) out vec4 fsColor;

mat3 sample3x3(vec2 uv) {
    float dx = 1.0 / textureSize(screenTexture, 0).x;
    float dy = 1.0 / textureSize(screenTexture, 0).y;

    return mat3(
        texture(screenTexture, uv + vec2(-dx, dy)).r,
        texture(screenTexture, uv + vec2(-dx, 0.0)).r,
        texture(screenTexture, uv + vec2(-dx, -dy)).r,
        texture(screenTexture, uv + vec2(0.0, dy)).r,
        texture(screenTexture, uv).r,
        texture(screenTexture, uv + vec2(0.0, -dy)).r,
        texture(screenTexture, uv + vec2(dx, dy)).r,
        texture(screenTexture, uv + vec2(dx, 0.0)).r,
        texture(screenTexture, uv + vec2(dx, -dy)).r);
}

float blur3x3(mat3 a) {
    // Gaussian 3x3 kernel.
    mat3 s = mat3(1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
        1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0,
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0);

    return s[0][0] * a[0][0] + s[0][1] * a[0][1] + s[0][2] * a[0][2] 
        + s[1][0] * a[1][0] + s[1][1] * a[1][1] + s[1][2] * a[1][2] 
        + s[2][0] * a[2][0] + s[2][1] * a[2][1] + s[2][2] * a[2][2];
}

float centralDifferences(mat3 a) {
    float gx = (a[1][2] - a[1][0]) * 0.5;
    float gy = (a[0][1] - a[2][1]) * 0.5;
    float magnitude = sqrt(gx * gx + gy * gy);
    return magnitude;
}

float contourLine(mat3 a) {
    float density = blur3x3(a);
    float densityGradient = centralDifferences(a);
    float contour = 0.0;
    for (int i = 0; i < contourIsoValueCount; ++i) {
        float delta = abs(density - contourIsoValues[i]);
        float distanceToContour = delta / densityGradient;

        //XXX: one must not violate smoothing kernel radius >= contour radius, otherwise one will see discontinuities.
        float size = min(contourSize, 1.5);
        contour = max(contour, 1.0 - smoothstep(size - 0.5, size + 0.5, distanceToContour));
    }
    return contour;
}

void main() {
    mat3 a = sample3x3(vsUV);

    vec4 valueColor = vec4(1.0, 0.0, 0.0, a[1][1]);//TODO: lookup!
    float contour = contourLine(a);

    fsColor = mix(valueColor, contourColor, contour);
}

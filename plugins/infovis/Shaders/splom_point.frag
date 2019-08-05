uniform int kernelType;

in float vsKernelSize;
in float vsPixelKernelSize;
in float vsValue;
in vec4 vsValueColor;

out vec4 fsColor;

// 2D circular kernel.
float circle2(vec2 p) {
    return length(p) < 0.5 ? 1.0 : 0.0;
}

// 2D gaussian kernel.
// Source: http://mathworld.wolfram.com/GaussianFunction.html
float gauss2(vec2 p) {
    const float sigma = 2.0f;
    return exp(-(0.5 * dot(p, p))) / (sigma * 2.506628274631000502415765284811);
}

void main(void) {
    const vec2 distance = gl_PointCoord.xy - vec2(0.5);
    float density;
    switch (kernelType) {
    case 0:
        density = circle2(distance);
        break;
    case 1:
        density = gauss2(distance * 6);
        break;
    default:
        density = 0.0;
    }

    const float attenuation = vsPixelKernelSize - vsKernelSize;
    density *= pow(1.0 - attenuation, 2);

    fsColor = toScreen(vsValue, vsValueColor, density);
}

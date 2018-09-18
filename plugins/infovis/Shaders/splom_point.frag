uniform float alphaScaling;
uniform int kernelType;

in vec4 vsColor;
in float vsKernelSize;
in float vsPixelKernelSize;

out vec4 fsColor;

// Hard 2D Circle.
float circle2(vec2 p) {
    return length(p) < 0.5 ? 1.0 : 0.0;
}

// 2D gaussian distribution.
// Source: http://mathworld.wolfram.com/GaussianFunction.html
float gauss2(vec2 p) {
    const float sigma = 2.0f;
    return exp(-(0.5 * dot(p, p))) / (sigma * 2.506628274631000502415765284811);
}

void main(void) {
    const vec2 distance = gl_PointCoord.xy - vec2(0.5);
    const float attenuation = vsPixelKernelSize - vsKernelSize;
    float alpha;
    switch (kernelType) {
    case 0:
        alpha = circle2(distance);
        break;
    case 1:
        alpha = gauss2(distance * 6);
        break;
    default:
        alpha = 0.0;
    }
    alpha *= pow(1.0 - attenuation, 2);
    alpha *= alphaScaling;

    // Blend against white.
    fsColor = vec4(vsColor.rgb, vsColor.a * alpha);
}

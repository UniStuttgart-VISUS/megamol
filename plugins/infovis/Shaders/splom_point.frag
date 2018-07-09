uniform float alphaScaling;

in vec4 vsColor;
in float vsKernelSize;
in float vsPixelKernelSize;

out vec4 fsColor;

// 2D gaussian distribution.
// Source: http://mathworld.wolfram.com/GaussianFunction.html
float gauss2(vec2 p) {
    const float sigma = 2.0f;
    return exp(-(0.5 * dot(p, p))) / (sigma * 2.506628274631000502415765284811);
}

void main(void) {
    const vec2 distance = gl_PointCoord.xy - vec2(0.5);
    const float attenuation = vsPixelKernelSize - vsKernelSize;
    float alpha = gauss2(distance * 6);
    alpha *= pow(1.0 - attenuation, 2);
    alpha *= alphaScaling;

    // Blend against white.
    fsColor = vec4(vsColor.rgb, vsColor.a * alpha);
}

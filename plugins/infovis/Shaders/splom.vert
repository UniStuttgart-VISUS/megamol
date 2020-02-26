//#define DEBUG_MINMAX 1

uniform vec4 viewport;
uniform mat4 modelViewProjection;

uniform int rowStride;

uniform float kernelWidth;
uniform bool attenuateSubpixel;

out float vsKernelSize;
out float vsPixelKernelSize;
out float vsValue;
out vec4 vsValueColor;
out vec4 vsPosition;

void main() {
    const Plot plot = plots[gl_InstanceID];
    const int rowOffset = gl_VertexID * rowStride;

    // Transform kernel size to screen space.
    const vec4 ndcKernelSize = modelViewProjection * vec4(kernelWidth, kernelWidth, 0.0, 0.0);
    const vec2 screenKernelSize = ndcKernelSize.xy * viewport.zw;
    vsKernelSize = max(screenKernelSize.x, screenKernelSize.y);

    if (attenuateSubpixel) {
        // Ensure a minimum pixel size to attenuate alpha depending on subpixels.
        vsPixelKernelSize = max(vsKernelSize, 1.0);
    } else {
        vsPixelKernelSize = vsKernelSize;
    }
    gl_PointSize = vsPixelKernelSize;

    if (valueColumn == -1) {
        vsValue = 1.0;
    } else {
        vsValue = normalizeValue(values[rowOffset + valueColumn]);
    }
    vsValueColor = flagifyColor(tflookup(vsValue), flags[gl_VertexID]);

    vsPosition = valuesToPosition(plot, 
        vec2(values[rowOffset + plot.indexX],
        values[rowOffset + plot.indexY]));
    if (bitflag_test(flags[gl_VertexID], FLAG_ENABLED | FLAG_FILTERED, FLAG_ENABLED)) {
    //if (true) {
        gl_Position = modelViewProjection * vsPosition;
		gl_ClipDistance[0] = 1.0;
    } else {
        gl_ClipDistance[0] = -1.0; // clipping cheat
    }
}

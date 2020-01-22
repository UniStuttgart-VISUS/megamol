layout(std430, binding = 0) buffer Histogram
{
    float histogram[];
};

layout(std430, binding = 1) buffer SelectedHistogram
{
    float selectedHistogram[];
};

uniform int binCount = 0;
uniform int colCount = 0;
uniform int logPlot = 0;
uniform float maxBinValue = 0.0;

uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);

out float binColor;
out float selection;

void main()
{
    int binId = gl_InstanceID / colCount;
    int colId = gl_InstanceID - colCount * binId; // integer modulo

    float histoVal = histogram[binId * colCount + colId];
    float selectedHistoVal = selectedHistogram[binId * colCount + colId];
    float maxHistoVal = maxBinValue;
    if (logPlot > 0) {
        histoVal = max(0.0, log(histoVal));
        selectedHistoVal = max(0.0, log(selectedHistoVal));
        maxHistoVal = max(1.0, log(maxHistoVal));
    }
    binColor = float(binId) / float(binCount - 1);

    vec3 pos = vec3(0.0);
    selection = 0.0;
    if (gl_VertexID == 0) { // bottom left
        pos = vec3(0.0, 0.0, 0.0);
    } else if (gl_VertexID == 1) { // bottom right
        pos = vec3(1.0, 0.0, 0.0);
    } else if (gl_VertexID == 2) { // top left
        pos = vec3(0.0, 1.0, 0.0);
        selection = histoVal / selectedHistoVal;
    } else if (gl_VertexID == 3) { // top right
        pos = vec3(1.0, 1.0, 0.0);
        selection = histoVal / selectedHistoVal;
    }

    float width = 10.0 / float(binCount);
    float height = 10.0 * histoVal / maxHistoVal;
    float posX = 12.0 * float(colId) + 1.0 + float(binId) * width;
    float posY = 2.0;

    pos.x = pos.x * width + posX;
    pos.y = pos.y * height + posY;

    gl_Position = projection * modelView * vec4(pos, 1.0);
}

layout(std430, binding = 0) buffer Histogram
{
    int histogram[];
};

layout(std430, binding = 1) buffer SelectedHistogram
{
    int selectedHistogram[];
};

layout(std430, binding = 2) buffer MaxBinValue
{
    int maxBinValue[];
};

uniform int binCount = 0;
uniform int colCount = 0;
uniform int logPlot = 0;

uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);

out float binColor;
out float selection;

void main()
{
    int binId = gl_InstanceID / colCount;
    int colId = gl_InstanceID - colCount * binId; // integer modulo

    float histoVal = float(histogram[binId * colCount + colId]);
    float selectedHistoVal = float(selectedHistogram[binId * colCount + colId]);
    float maxHistoVal = float(maxBinValue[0]);
    if (logPlot > 0) {
        histoVal = max(0.0, log(histoVal));
        selectedHistoVal = max(0.0, log(selectedHistoVal));
        maxHistoVal = max(1.0, log(maxHistoVal));
    }
    binColor = float(binId) / float(binCount - 1);

    float width = 10.0 / float(binCount);
    float height = 10.0 * histoVal / maxHistoVal;
    float posX = 12.0 * float(colId) + 1.0 + float(binId) * width;
    float posY = 2.0;

    vec2 pos = vec2(0.0);
    selection = 0.0;
    if (gl_VertexID == 0) { // bottom left
        pos = vec2(posX, posY);
    } else if (gl_VertexID == 1) { // bottom right
        pos = vec2(posX + width, posY);
    } else if (gl_VertexID == 2) { // top left
        pos = vec2(posX, posY + height);
        selection = histoVal / selectedHistoVal;
    } else if (gl_VertexID == 3) { // top right
        pos = vec2(posX + width, posY + height);
        selection = histoVal / selectedHistoVal;
    }

    gl_Position = projection * modelView * vec4(pos, 0.0, 1.0);
}

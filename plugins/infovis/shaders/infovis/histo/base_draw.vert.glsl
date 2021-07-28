#version 430

#include "common.inc.glsl"

uniform uint maxBinValue = 1;
uniform int logPlot = 0;

uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);

out float binColor;
out float selection;

void main() {
    int binId = gl_InstanceID / int(numCols);
    int colId = gl_InstanceID - int(numCols) * binId; // integer modulo

    float histoVal = float(histogram[binId * numCols + colId]);
    float selectedHistoVal = float(selectedHistogram[binId * numCols + colId]);
    float maxHistoVal = float(maxBinValue);
    if (logPlot > 0) {
        histoVal = max(0.0, log(histoVal));
        selectedHistoVal = max(0.0, log(selectedHistoVal));
        maxHistoVal = max(1.0, log(maxHistoVal));
    }
    binColor = float(binId) / float(numBins - 1);

    float width = 10.0 / float(numBins);
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

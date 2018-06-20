struct Plot {
    uint indexX;
    uint indexY;
    float offsetX;
    float offsetY;
    float sizeX;
    float sizeY;
    float minX;
    float minY;
    float maxX;
    float maxY;
};

layout(std430, binding = 2) buffer PlotSSBO {
    Plot plots[];
};

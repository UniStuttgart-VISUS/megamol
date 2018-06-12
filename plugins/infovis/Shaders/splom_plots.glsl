struct Plot {
    uint indexX;
    uint indexY;
    float offsetX;
    float offsetY;
    float sizeX;
    float sizeY;
    float minX;
    float maxX;
    float minY;
    float maxY;
};

layout(std430, shared, binding = 2) buffer PlotSSBO {
    Plot plots[];
};

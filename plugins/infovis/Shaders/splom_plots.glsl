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
    float smallTickX;
    float smallTickY;
};

layout(std430, binding = 2) buffer PlotSSBO {
    Plot plots[];
};

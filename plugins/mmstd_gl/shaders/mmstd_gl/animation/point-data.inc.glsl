struct point {
    float x;
    float y;
    float z;
    // packed unorm 4x8
    uint col;
};

layout(std430, binding = 1) buffer pointData {
    point points[];
};

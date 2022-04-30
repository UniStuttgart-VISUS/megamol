struct pos {
    float x;
    float y;
    float z;
};

struct quat {
    float x;
    float y;
    float z;
    float w;
};

layout(std430, binding = 0) buffer pos_data {
    pos pos_array[];
};
layout(std430, binding = 1) buffer quat_data {
    quat quat_array[];
};
layout(std430, binding = 2) buffer rad_data {
    pos rad_array[];
};
layout(std430, binding = 3) buffer col_data {
    quat col_array[];
};

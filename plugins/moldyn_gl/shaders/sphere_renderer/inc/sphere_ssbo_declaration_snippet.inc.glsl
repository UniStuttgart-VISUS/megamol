
struct SphereParams {
    float posX; float posY; float posZ; float posR;
    float r; float g; float b; float a;
};
layout(std430, binding = 3) buffer shader_data {
    SphereParams theBuffer[];
};

layout(std430, binding = 2) buffer SpherePos {
    vec4 inPosition[];
};
layout(std430, binding = 3) buffer SphereColor {
    vec4 inColor[];
};

void access_data(uint idx, out vec3 objPos, out vec4 objColor, out float rad) {
    objPos = inPosition[idx].xyz;

    if (useGlobalRad) {
        rad = globalRad;
    } else {
        rad = inPosition[idx].w;
    }

    if (useGlobalCol) {
        objColor = globalCol;
    } else {
        objColor = inColor[idx];
    }
}

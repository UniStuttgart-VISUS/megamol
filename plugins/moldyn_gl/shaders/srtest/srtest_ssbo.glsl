layout(std430, binding = 2) buffer SpherePos {
    vec4 inPosition[];
};
layout(std430, binding = 3) buffer SphereColor {
    vec4 inColor[];
};

void access_data(out vec3 objPos, out vec4 objColor, out float rad) {
    objPos = inPosition[gl_VertexID].xyz;

    if (useGlobalRad) {
        rad = globalRad;
    } else {
        rad = inPosition[gl_VertexID].w;
    }

    if (useGlobalCol) {
        objColor = globalCol;
    } else {
        objColor = inColor[gl_VertexID];
    }
}

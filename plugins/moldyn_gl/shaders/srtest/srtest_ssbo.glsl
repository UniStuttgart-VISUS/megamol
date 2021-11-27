/*layout(std430, binding = 2) buffer SpherePos {
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
}*/

layout(std430, binding = 2) buffer X {
    float xPtr[];
};
layout(std430, binding = 3) buffer Y {
    float yPtr[];
};
layout(std430, binding = 4) buffer Z {
    float zPtr[];
};
layout(std430, binding = 5) buffer RAD {
    float radPtr[];
};
layout(std430, binding = 6) buffer R {
    float rPtr[];
};
layout(std430, binding = 7) buffer G {
    float gPtr[];
};
layout(std430, binding = 8) buffer B {
    float bPtr[];
};
layout(std430, binding = 9) buffer A {
    float aPtr[];
};

void access_data(uint idx, out vec3 objPos, out vec4 objColor, out float rad) {
    objPos = vec3(xPtr[idx], yPtr[idx], zPtr[idx]);

    if (useGlobalRad) {
        rad = globalRad;
    } else {
        rad = radPtr[idx];
    }

    if (useGlobalCol) {
        objColor = globalCol;
    } else {
        objColor = vec4(rPtr[idx], gPtr[idx], bPtr[idx], aPtr[idx]);
    }
}

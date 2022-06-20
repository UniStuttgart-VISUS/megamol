#ifdef __SRTEST_UPLOAD_POS_COL_SEP__
layout(std430, binding = 2) readonly buffer SpherePos {
    vec4 inPosition[];
};
layout(std430, binding = 3) readonly buffer SphereColor {
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
#elif defined(__SRTEST_UPLOAD_VEC3_SEP__)
struct ppPoint {
    float x;
    float y;
    float z;
};

layout(std430, binding = 2) readonly buffer SpherePos {
    //vec3 inPosition[];
    ppPoint pos[];
};
layout(std430, binding = 3) readonly buffer SphereColor {
    uint inColor[];
};

void access_data(uint idx, out vec3 objPos, out vec4 objColor, out float rad) {
    //objPos = inPosition[idx];
    objPos = vec3(pos[idx].x, pos[idx].y, pos[idx].z);

    if (useGlobalRad) {
        rad = globalRad;
    } else {
        //rad = 0.001f;
        rad = 0.5f;
    }

    if (useGlobalCol) {
        objColor = globalCol;
    } else {
        objColor = unpackUnorm4x8(inColor[idx]);
    }
}
#elif defined(__SRTEST_UPLOAD_NO_SEP__) || defined(__SRTEST_UPLOAD_BUFFER_ARRAY__)
struct ppPoint {
    vec3 pos;
    uint col;
};
layout(std430, binding = 2) readonly buffer SpherePos {
    ppPoint pos_col[];
};

void access_data(uint idx, out vec3 objPos, out vec4 objColor, out float rad) {
    objPos = pos_col[idx].pos;

    if (useGlobalRad) {
        rad = globalRad;
    } else {
        //rad = 0.001f;
        rad = 0.5f;
    }

    if (useGlobalCol) {
        objColor = globalCol;
    } else {
        objColor = unpackUnorm4x8(pos_col[idx].col);
    }
}
#elif defined(__SRTEST_UPLOAD_FULL_SEP__) || defined(__SRTEST_UPLOAD_COPY_IN__)
layout(std430, binding = 2) readonly buffer X {
    float xPtr[];
};
layout(std430, binding = 3) readonly buffer Y {
    float yPtr[];
};
layout(std430, binding = 4) readonly buffer Z {
    float zPtr[];
};
layout(std430, binding = 5) readonly buffer RAD {
    float radPtr[];
};
layout(std430, binding = 6) readonly buffer R {
    float rPtr[];
};
layout(std430, binding = 7) readonly buffer G {
    float gPtr[];
};
layout(std430, binding = 8) readonly buffer B {
    float bPtr[];
};
layout(std430, binding = 9) readonly buffer A {
    float aPtr[];
};
#ifdef __SRTEST_UPLOAD_FULL_SEP__
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
#endif
#endif

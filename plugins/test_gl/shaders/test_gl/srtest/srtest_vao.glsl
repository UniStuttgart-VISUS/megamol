layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;

void access_data(uint idx, out vec3 objPos, out vec4 objColor, out float rad) {
    objPos = inPosition.xyz;

    if (useGlobalRad) {
        rad = globalRad;
    } else {
        rad = inPosition.w;
    }

    if (useGlobalCol) {
        objColor = globalCol;
    } else {
        objColor = inColor;
    }
}

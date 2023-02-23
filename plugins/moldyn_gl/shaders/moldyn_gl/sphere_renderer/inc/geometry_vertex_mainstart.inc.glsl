
layout (location = 0) in vec4 inPosition;
layout (location = 1) in vec4 inColor;
layout (location = 2) in float inColIdx;

out vec4 colorgs;
out float colidxgs;

void main(void) {

    colorgs = inColor;
    colidxgs = inColIdx;
    gl_Position = inPosition;

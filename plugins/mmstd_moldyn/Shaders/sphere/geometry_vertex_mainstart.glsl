
in vec4 inPosition;
in vec4 inColor;
in float inColIdx;

out vec4 colorgs;
out float colidxgs;

void main(void) {

    colorgs = inColor;
    colidxgs = inColIdx;
    gl_Position = inPosition;

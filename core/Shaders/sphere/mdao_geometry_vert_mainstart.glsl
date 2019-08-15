
in vec4 inPosition;
in vec4 inColor;
in float inColIdx;

out vec4 colorgs;
out float colidxgs;

void main(void) {
    colorgs = inColor;
    colidxgs = inColIdx;

    if (!(bool(flagsAvailable)) || (bool(flagsAvailable) && bitflag_isVisible(flag))) {
        // Set gl_Position depending on flags

        gl_Position = inPosition;
    }

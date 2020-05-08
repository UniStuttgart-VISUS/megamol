
uniform int instanceOffset;

// Only used by SPLAT render mode:
uniform int attenuateSubpixel;
out float effectiveDiameter;

void main(void) {

    float inColIdx;
    vec4 inColor;
    vec4 inPosition;

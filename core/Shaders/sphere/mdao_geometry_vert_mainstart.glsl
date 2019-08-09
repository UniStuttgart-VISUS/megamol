in vec4 position;
in vec4 color;

out vec4 colorgs;

uniform vec4 viewAttr;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

uniform mat4 MVP;
uniform mat4 MVPinv;

uniform float inGlobalRadius;
uniform bool inUseGlobalColor;
uniform vec4 inGlobalColor;

uniform bool inUseTransferFunction;
uniform sampler1D inTransferFunction;
uniform vec2 inIndexRange;

uniform vec4 clipDat;
uniform vec4 clipCol;

void main(void) {
    colorgs = color;
    gl_Position = position;

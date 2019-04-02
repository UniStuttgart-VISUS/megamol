in vec4 position;
in vec4 color;

out vec4 colorgs;

uniform vec4 inViewAttr;
uniform vec3 inCamFront;
uniform vec3 inCamUp;
uniform vec3 inCamRight;
uniform vec4 inCamPos;

uniform mat4 inMvp;
uniform mat4 inMvpInverse;

uniform float inGlobalRadius;
uniform bool inUseGlobalColor;
uniform vec4 inGlobalColor;

uniform bool inUseTransferFunction;
uniform sampler1D inTransferFunction;
uniform vec2 inIndexRange;

uniform vec4 inClipDat;
uniform vec4 inClipCol;

void main(void) {
    colorgs = color;
    gl_Position = position;
}
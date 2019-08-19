
in vec4 objPos;
in vec4 camPos;
in float squarRad;
in float rad;
in vec4 vertColor;

uniform vec4 viewAttr;

uniform mat4 MVPinv;
uniform mat4 MVPtransp;

uniform vec4 clipDat;
uniform vec4 clipCol;

uniform bool inUseHighPrecision;

out vec4 outColor;
out vec4 outNormal;

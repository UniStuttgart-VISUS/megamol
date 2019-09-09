#extension GL_ARB_explicit_attrib_location : require   // glsl version 130

uniform vec4 viewAttr; // TODO: check fragment position if viewport starts not in (0, 0)

uniform mat4 MVPinv;
uniform mat4 MVPtransp;

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;

in vec4 vertColor;
in vec4 objPos;
in vec4 camPos;
in vec4 outlightDir;
in float rad;
in float squarRad;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

layout(location = 0) out vec4 outColor;

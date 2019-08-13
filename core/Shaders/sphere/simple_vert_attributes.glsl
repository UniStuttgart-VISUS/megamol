
in vec4 inPosition;
in vec4 inColor;
in float inColIdx;

out vec4 objPos;
out vec4 camPos;
out vec4 lightPos;
out float squarRad;
out float rad;
out vec4 vertColor;

#ifdef DEFERRED_SHADING
out float pointSize;
#endif // DEFERRED_SHADING

#ifdef RETICLE
out vec2 centerFragment;
#endif // RETICLE

uniform vec4 viewAttr;
uniform vec4 lpos;

#ifdef WITH_SCALING
uniform float scaling;
#endif // WITH_SCALING

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;

uniform mat4 MVinv;
uniform mat4 MVP;

uniform float constRad;
uniform vec4 globalCol;
uniform int useGlobalCol;
uniform int useTf;


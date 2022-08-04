
out vec4 objPos;
out vec4 camPos;
out vec4 outlightDir;
out float squarRad;
out float rad;
out vec4 vertColor;

#ifdef DEFERRED_SHADING
out float pointSize;
#endif // DEFERRED_SHADING

// OUTLINE / ifdef RETICLE
out float sphere_frag_radius;
out vec2 sphere_frag_center;
// OUTLINE
uniform float outlineWidth = 0.0;

uniform vec4 viewAttr;
uniform vec4 lightDir;

#ifdef WITH_SCALING
uniform float scaling;
#endif // WITH_SCALING

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

uniform vec4 clipDat;
uniform vec4 clipCol;

uniform mat4 MVinv;
uniform mat4 MVtransp;
uniform mat4 MVP;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;

uniform float constRad;
uniform vec4 globalCol;
uniform int useGlobalCol;
uniform int useTf;

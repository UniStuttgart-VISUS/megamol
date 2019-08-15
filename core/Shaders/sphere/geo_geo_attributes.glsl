
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 colorgs[1];
in float colidxgs[1];

out vec4 vertColor;
out vec4 objPos;
out vec4 camPos;
out vec4 lightPos;
out float rad;
out float squarRad;

#ifdef WITH_SCALING
uniform float scaling;
#endif // WITH_SCALING

uniform vec4 viewAttr; // TODO: check fragment position if viewport starts not in (0, 0)
uniform vec4 lpos;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

uniform mat4 MVP;
uniform mat4 MVinv;

uniform float constRad;
uniform vec4 globalCol;
uniform int useGlobalCol;
uniform int useTf;

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;

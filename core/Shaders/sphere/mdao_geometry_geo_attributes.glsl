
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 colorgs[1];
in float colidxgs[1];

out vec4 vertColor;
out vec4 objPos;
out vec4 camPos;
out float squarRad;
out float rad;

#ifdef WITH_SCALING
uniform float scaling;
#endif // WITH_SCALING

uniform vec4 viewAttr;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

uniform mat4 MVP;
uniform mat4 MVinv;

uniform vec4 clipDat;
uniform vec4 clipCol;

uniform vec4 inConsts1;
uniform sampler1D tfTexture;
uniform vec4 globalCol;
uniform int useGlobalCol;
uniform int useTf;

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w

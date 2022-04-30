uniform vec4 viewAttr;

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
uniform vec3 clipCol;

uniform vec4 inConsts1;
attribute float colIdx;
uniform sampler1D colTab;

uniform sampler2D depthTex;
uniform ivec3 depthTexParams;
#define DEPTHMIP_WIDTH depthTexParams.x
#define DEPTHMIP_HEIGHT depthTexParams.y
#define DEPTHMIP_MAXLEVEL depthTexParams.z
// new camera
uniform mat4 mv_inv;
uniform mat4 mv_inv_transp;
uniform mat4 mvp;
uniform mat4 mvp_inv;
uniform mat4 mvp_transp;
uniform vec4 light_dir;
uniform vec4 cam_pos;
// end new camera
varying vec4 objPos;
varying vec4 camPos;
varying vec4 lightDir;
varying float squarRad;
varying float rad;

#ifdef RETICLE
varying vec2 centerFragment;
#endif // RETICLE

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w

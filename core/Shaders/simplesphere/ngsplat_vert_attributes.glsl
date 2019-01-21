#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_EXT_gpu_shader4 : require
uniform vec4 viewAttr;

uniform float scaling;
uniform float alphaScaling;
uniform float zNear;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
uniform mat4 modelViewProjection;
uniform mat4 modelViewInverse;
#endif // CALC_CAM_SYS

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;
uniform int instanceOffset;
uniform int attenuateSubpixel;

uniform vec4 inConsts1;
in float colIdx;
uniform sampler1D colTab;

out vec4 objPos;
out vec4 camPos;
//varying vec4 lightPos;
out float squarRad;
out float rad;
out float effectiveDiameter;

#ifdef DEFERRED_SHADING
out float pointSize;
#endif

#ifdef RETICLE
out vec2 centerFragment;
#endif // RETICLE

out vec4 vsColor;

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w
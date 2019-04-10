#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_EXT_gpu_shader4 : require

uniform vec4 viewAttr;
uniform float scaling;
uniform vec4 lpos;

uniform float alphaScaling;
uniform float zNear;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;
uniform int instanceOffset;

uniform int attenuateSubpixel;

uniform mat4 MVP;
uniform mat4 MVinv;

uniform vec4 inConsts1;
uniform sampler1D colTab;
uniform vec4 globalCol;

out vec4 objPos;
out vec4 camPos;
out vec4 lightPos;
out float squarRad;
out float rad;
out vec4 vertColor;

out float effectiveDiameter;

#ifdef DEFERRED_SHADING
out float pointSize;
#endif // DEFERRED_SHADING

#ifdef RETICLE
out vec2 centerFragment;
#endif // RETICLE

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w
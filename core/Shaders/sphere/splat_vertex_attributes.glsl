#extension GL_ARB_shader_storage_buffer_object : require   // glsl version 430
#extension GL_EXT_gpu_shader4 : require                    // glsl version 130

uniform vec4 viewAttr;
uniform float scaling;
uniform vec4 lpos;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;
uniform int instanceOffset;

uniform mat4 MVinv;
uniform mat4 MVP;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;

uniform float constRad;
uniform vec4 globalCol;
uniform int useGlobalCol;
uniform int useTf;

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

uniform int attenuateSubpixel;
out float effectiveDiameter;
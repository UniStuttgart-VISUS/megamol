#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_EXT_gpu_shader4 : require
uniform vec4 viewAttr;

uniform float scaling;

uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;
uniform int instanceOffset;

uniform mat4 MVinv;
uniform mat4 MVP;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;

uniform vec4 inConsts1;
uniform sampler1D colTab;

out vec4 objPos;
out vec4 camPos;
out vec4 lightPos;
out float squarRad;
out float rad;
out vec4 vertColor;

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w

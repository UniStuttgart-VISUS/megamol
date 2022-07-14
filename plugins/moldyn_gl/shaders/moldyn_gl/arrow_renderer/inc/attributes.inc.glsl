uniform vec4 viewAttr;

uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;

uniform mat4 MVinv;
uniform mat4 MVtransp;
uniform mat4 MVP;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;

uniform vec4 lightDir;

uniform vec4 inConsts1;
uniform sampler1D colTab;
uniform float lengthScale;
uniform float lengthFilter;
uniform uint flagsAvailable;

uniform vec4 clipDat;
uniform vec4 clipCol;

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w

uniform vec4 inConsts1;
uniform sampler1D colTab;

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;

in vec4 inVertex;
in vec4 inColor;
in float colIdx;

out vec4 vertColor;

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w

//#define FOGGING_SES
//#define FLATSHADE_SES
#define OGL_DEPTH_SES

//#define SET_COLOR
//#define COLOR1 vec3( 249.0/255.0, 187.0/255.0, 103.0/255.0)
#define COLOR1 vec3( 183.0/255.0, 204.0/255.0, 220.0/255.0)

//#define COLOR_SES
#define COLOR_BLUE   vec3( 145.0/255.0, 191.0/255.0, 219.0/255.0)
#define COLOR_GREEN  vec3( 161.0/255.0, 215.0/255.0, 106.0/255.0)
#define COLOR_YELLOW vec3( 255.0/255.0, 255.0/255.0, 191.0/255.0)
#define COLOR_RED    vec3( 228.0/255.0, 37.0/255.0, 34.0/255.0)
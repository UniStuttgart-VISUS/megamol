//#define FOGGING_SES
//#define FLATSHADE_SES
#define OGL_DEPTH_SES
//#define SFB_DEMO

#define CLIP
#define DEPTH
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

#ifdef DEBUG
#undef CLIP
#define RETICLE
#define AXISHINTS
#endif // DEBUG

//#define BULLSHIT

#ifndef FLACH
#define FLACH
#endif

//#define SET_COLOR
//#define COLOR1 vec3( 249.0/255.0, 187.0/255.0, 103.0/255.0)
#define COLOR1 vec3( 183.0/255.0, 204.0/255.0, 220.0/255.0)

//#define COLOR_SES
//#define COLOR_BLUE vec3( 145.0/255.0, 191.0/255.0, 219.0/255.0)
#define COLOR_BLUE vec3( 0.0/255.0, 173.0/255.0, 238.0/255.0)
//#define COLOR_GREEN vec3( 161.0/255.0, 215.0/255.0, 106.0/255.0)
//#define COLOR_GREEN vec3( 236.0/255.0, 28.0/255.0, 36.0/255.0) // actually red...
#define COLOR_GREEN vec3( 0.0/255.0, 165.0/255.0, 81.0/255.0)
//#define COLOR_YELLOW vec3( 255.0/255.0, 255.0/255.0, 191.0/255.0)
#define COLOR_YELLOW vec3( 255.0/255.0, 221.0/255.0, 21.0/255.0)
#define COLOR_RED vec3( 228.0/255.0, 37.0/255.0, 34.0/255.0)

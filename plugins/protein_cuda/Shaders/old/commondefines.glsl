//#define FOGGING_SES
//#define FLATSHADE_SES

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

#define CLIP
#define DEPTH
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

//#define RETICLE
//#define DISCARD_COLOR_MARKER
#ifdef DEBUG
#undef CLIP
#define RETICLE
#define AXISHINTS
#endif // DEBUG

//#define BULLSHIT

#ifndef FLACH
#define FLACH
#endif

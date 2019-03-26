#define CLIP
#define DEPTH
#define WITH_SCALING
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

#ifdef DEBUG
#undef CLIP
#define RETICLE
#define AXISHINTS
#endif // DEBUG

#ifndef FLACH
#define FLACH
#endif // FLACH

#define CLIP
#define DEPTH
#define WITH_SCALING
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

//#define HALO
#ifdef HALO
    #define HALO_RAD 3.0
#endif // HALO

//#define DEBUG
#ifdef DEBUG
    #undef CLIP
    #define RETICLE
    #define AXISHINTS
#endif // DEBUG

#ifndef FLACH
    #define FLACH
#endif // FLACH

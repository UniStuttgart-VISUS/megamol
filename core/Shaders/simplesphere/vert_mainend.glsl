#ifdef SMALL_SPRITE_LIGHTING
    // for normal crowbaring on very small sprites
    lightPos.w = (clamp(gl_PointSize, 1.0, 5.0) - 1.0) / 4.0;
#else
    lightPos.w = 1.0;
#endif // SMALL_SPRITE_LIGHTING
    
#ifdef RETICLE
    centerFragment = gl_Position.xy / gl_Position.w;
#endif // RETICLE

    // gl_PointSize = 32.0;
}

#ifdef SMALL_SPRITE_LIGHTING
    // for normal crowbaring on very small sprites
    outLightPos.w = (clamp(gl_PointSize, 1.0, 5.0) - 1.0) / 4.0;
#else
    outLightPos.w = 1.0;
#endif // SMALL_SPRITE_LIGHTING
    
#ifdef RETICLE
    centerFragment = gl_Position.xy / gl_Position.w;
#endif // RETICLE

#ifdef DEFERRED_SHADING
    pointSize = gl_PointSize;
#endif // DEFERRED_SHADING

    // gl_PointSize = 32.0;
}

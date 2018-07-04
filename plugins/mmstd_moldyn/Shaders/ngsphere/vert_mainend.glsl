#ifdef SMALL_SPRITE_LIGHTING
    // for normal crowbaring on very small sprites
    lightPos.w = (clamp(gl_PointSize, 1.0, 5.0) - 1.0) / 4.0;
#else
    lightPos.w = 1.0;
#endif // SMALL_SPRITE_LIGHTING
    
#ifdef RETICLE
    centerFragment = gl_Position.xy / gl_Position.w;
#endif // RETICLE
    //gl_Position = MVP * vec4(inPos.xyz, 1.0);
    //gl_Position /= gl_Position.w;
    //gl_PointSize = 8.0;

}


    // Debug
    // gl_PointSize = 32.0;

#ifdef SMALL_SPRITE_LIGHTING
    // for normal crowbaring on very small sprites
    outlightDir.w = (clamp(gl_PointSize, 1.0, 5.0) - 1.0) / 4.0;
#else
    outlightDir.w = 1.0;
#endif // SMALL_SPRITE_LIGHTING

    sphere_frag_radius = gl_PointSize / 2.0;
    gl_PointSize += (2.0 * outlineWidth);
    sphere_frag_center = ((gl_Position.xy / gl_Position.w) - vec2(-1.0, -1.0)) / vec2(viewAttr.z, viewAttr.w);

#ifdef DEFERRED_SHADING
    pointSize = gl_PointSize;
#endif // DEFERRED_SHADING

}

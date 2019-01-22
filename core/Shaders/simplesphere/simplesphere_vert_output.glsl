    vec2 screen = vec2(right.x / right.w - left.x / left.w, top.y / top.w - bottom.y / bottom.w) * 0.5 * winHalf;
    gl_PointSize = max(screen.x, screen.y) + 0.5;
    
    vec2 mid = vec2(right.x / right.w + left.x / left.w, top.y / top.w + bottom.y / bottom.w);
    mid.xy *= 0.5;
    gl_Position = vec4(mid.xy, (front.z / front.w), (od > clipDat.w) ? 0.0 : 1.0);

#ifdef DEFERRED_SHADING
    pointSize = gl_PointSize;
#endif // DEFERRED_SHADING

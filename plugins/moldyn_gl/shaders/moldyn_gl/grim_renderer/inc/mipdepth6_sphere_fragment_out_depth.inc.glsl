
// calculate depth
#ifdef DEPTH
    vec4 Ding = vec4(sphereintersection + objPos.xyz, 1.0);
    float depth = dot(mvp_transp[2], Ding);
    float depthW = dot(mvp_transp[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
#ifndef CLIP
    gl_FragDepth = (radicand < 0.0) ? 1.0 : ((depth / depthW) + 1.0) * 0.5;
    gl_FragColor.rgb = (radicand < 0.0) ? gl_Color.rgb : gl_FragColor.rgb;
#endif // CLIP
#ifdef DISCARD_COLOR_MARKER
    Ding = vec4(objPos.xyz, 1.0);
    depth = dot(mvp_transp[2], Ding);
    depthW = dot(mvp_transp[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
#endif // DISCARD_COLOR_MARKER
#endif // DEPTH

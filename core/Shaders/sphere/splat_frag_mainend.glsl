
    //gl_FragColor = vec4((gl_PointCoord.xy - vec2(0.5)) * vec2(2.0), 0.0, 1.0);
    //gl_FragColor = vec4(gl_PointCoord.xy, 0.5, 1.0);
    //gl_FragColor = vertColor;
    vec2 dist = gl_PointCoord.xy - vec2(0.5);
    float d = sqrt(dot(dist, dist));
    float alpha = 0.5-d;
    alpha *= effectiveDiameter * effectiveDiameter;
    alpha *= alphaScaling;
    //alpha = 0.5;
#if 0
    // blend against white!
    outColor = vec4(vertColor.rgb, alpha);
#else
    outColor = vec4(vertColor.rgb * alpha, alpha);
#endif
    //outColor = vec4(vertColor.rgb, 1.0);
}

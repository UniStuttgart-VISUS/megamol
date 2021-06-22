
#ifdef RETICLE
    if (min(abs(gl_FragCoord.x - centerFragment.x), abs(gl_FragCoord.y - centerFragment.y)) < 2.0f) {
        //outColor.rgb = vec3(1.0, 1.0, 0.5);
        outColor.rgb += vec3(0.3, 0.3, 0.5);
    }
#endif // RETICLE

    //outColor.rgb = normal;
    //outColor = vec4(1.0, 0.0, 0.0, 1.0);
}

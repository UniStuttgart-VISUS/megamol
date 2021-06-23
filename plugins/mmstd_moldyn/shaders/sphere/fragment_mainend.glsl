
#ifdef RETICLE
    if (min(abs(gl_FragCoord.x - sphere_frag_center.x), abs(gl_FragCoord.y - sphere_frag_center.y)) < 2.0f) {
        //outColor.rgb = vec3(1.0, 1.0, 0.5);
        outColor.rgb += vec3(0.3, 0.3, 0.5);
    }
#endif // RETICLE

    //outColor.rgb = normal;
    //outColor = vec4(1.0, 0.0, 0.0, 1.0);
}

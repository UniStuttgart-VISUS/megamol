#ifdef RETICLE
    coord = gl_FragCoord
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);
    if (min(abs(coord.x - centerFragment.x), abs(coord.y - centerFragment.y)) < 0.002) {
        //gl_FragColor.rgb = vec3(1.0, 1.0, 0.5);
        gl_FragColor.rgb += vec3(0.3, 0.3, 0.5);
    }
#endif // RETICLE
//    gl_FragColor.rgb = normal;
}

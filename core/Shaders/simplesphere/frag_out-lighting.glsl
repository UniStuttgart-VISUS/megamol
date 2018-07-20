    // phong lighting with directional light
    gl_FragColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color.rgb), color.a);
    //gl_FragColor = color;

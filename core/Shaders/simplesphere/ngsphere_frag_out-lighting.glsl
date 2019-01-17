    // phong lighting with directional light
    outCol = vec4(LocalLighting(ray, normal, lightPos.xyz, color.rgb), color.a);
    //outCol = color;
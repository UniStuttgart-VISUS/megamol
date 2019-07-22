    // phong lighting with directional light
    outColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color.rgb), color.a);
    //outColor = color;

 
    // Phong lighting with directional light
    outColor = vec4(LocalLighting(ray, normal, outlightDir.xyz, color.rgb), color.a);
    //outColor = color;

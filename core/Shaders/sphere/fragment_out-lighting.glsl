 
    // Phong lighting with directional light
    outColor = vec4(LocalLighting(ray, normal, outLightPos.xyz, color.rgb), color.a);
    //outColor = color;

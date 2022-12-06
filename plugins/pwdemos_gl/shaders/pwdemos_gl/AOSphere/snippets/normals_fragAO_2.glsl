        gl_FragColor.xyz = vec3(aoFactor);
        //float grey = ((depth / depthW) + 1.0) * 0.5;
        //vec3 camIn = (gl_ModelViewMatrixInverse[3] + gl_ModelViewMatrixInverse[2]).xyz;
        //camIn = normalize(camIn);
        //float grey = abs(dot(normal, camIn));
        //gl_FragColor = vec4(vec3(grey), 1.0);

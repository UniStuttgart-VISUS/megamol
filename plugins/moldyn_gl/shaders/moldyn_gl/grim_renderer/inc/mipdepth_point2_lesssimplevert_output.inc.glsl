
    gl_Position = vec4((mins + maxs) * 0.5, 0.0, (od > clipDat.w) ? 0.0 : 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    //gl_PointSize = max(maxs.x, maxs.y) + 0.5; // diameter

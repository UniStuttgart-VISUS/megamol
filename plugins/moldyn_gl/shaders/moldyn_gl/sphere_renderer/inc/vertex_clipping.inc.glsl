    
    // Clipping
    float od = clipDat.w - 1.0;
    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        od = dot(objPos.xyz, clipDat.xyz) - rad;
    }

    gl_ClipDistance[0] = 1.0;
    if (od > clipDat.w)  {
        gl_ClipDistance[0] = -1.0;
        vertColor = clipCol;
    }

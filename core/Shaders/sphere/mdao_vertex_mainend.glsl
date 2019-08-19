
    if (!(bool(flagsAvailable)) || (bool(flagsAvailable) && bitflag_isVisible(flag))) {
        // Set gl_Position depending on flags
        
        // Clipping
        float od = clipDat.w - 1.0;
        if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
            od = dot(objPos.xyz, clipDat.xyz) - rad;
        }
        
        // Outgoing vertex position and point size
        gl_Position = vec4((mins + maxs) * 0.5, projPos.z, (od > clipDat.w) ? 0.0 : 1.0);
        maxs = (maxs - mins) * 0.5 * winHalf;
        gl_PointSize = max(maxs.x, maxs.y) + 0.5;
    }

}

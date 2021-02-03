 
    // Set gl_Position depending on flags (no fragment test required for visibility test)
    if (!(bool(flags_enabled)) || (bool(flags_enabled) && bitflag_isVisible(flag))) {
        
        // Outgoing vertex position and point size
        gl_Position = vec4((mins + maxs) * 0.5, projPos.z, 1.0);
        maxs = (maxs - mins) * 0.5 * winHalf;
        gl_PointSize = max(maxs.x, maxs.y) + 0.5;

    } else {
        gl_ClipDistance[0] = -1.0;
    }

    // object pivot point in object space
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)

    // calculate cam position
    camPos = cam_pos; // (C) by Christoph
    camPos.xyz -= objPos.xyz; // cam pos to glyph space

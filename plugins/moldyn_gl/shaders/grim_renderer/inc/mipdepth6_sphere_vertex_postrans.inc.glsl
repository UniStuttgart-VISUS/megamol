    // object pivot point in object space
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)

    // calculate cam position
    camPos = cam_pos;
    camPos.xyz -= objPos.xyz; // cam pos to glyph space

    // calculate light position in glyph space
    lightDir = light_dir;

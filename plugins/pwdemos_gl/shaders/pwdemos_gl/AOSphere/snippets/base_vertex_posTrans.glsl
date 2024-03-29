    // object pivot point in object space
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)

    // calculate cam position
    camPos = gl_ModelViewMatrixInverse[3]; // (C) by Christoph
    camPos.xyz -= objPos.xyz; // cam pos to glyph space

    // calculate light position in glyph space
    lightPos = gl_ModelViewMatrixInverse * gl_LightSource[0].position;

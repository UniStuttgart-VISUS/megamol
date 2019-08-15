
    // Color  
    vertColor = inColor;
    if (bool(useGlobalCol))  {
        vertColor = globalCol;
    }
    if (bool(useTf)) {
        vertColor = tflookup(inColIdx);
    }
    // Overwrite color depending on flags
    if (bool(flagsAvailable)) {
        if (bitflag_test(flag, FLAG_SELECTED, FLAG_SELECTED)) {
            vertColor = flagSelectedCol;
        }
        if (bitflag_test(flag, FLAG_SOFTSELECTED, FLAG_SOFTSELECTED)) {
            vertColor = flagSoftSelectedCol;
        }
        //if (!bitflag_isVisible(flag)) {
        //    vertColor = vec4(0.0, 0.0, 0.0, 0.0);
        //}
    }
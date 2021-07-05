
    // Color  
    vertColor = inColor;
    if (bool(useGlobalCol))  {
        vertColor = globalCol;
    }
    if (bool(useTf)) {
        vertColor = tflookup(inColIdx);
    }
    // Overwrite color depending on flags
    if (bool(flags_enabled)) {
        if (bitflag_test(flag, FLAG_SELECTED, FLAG_SELECTED)) {
            vertColor = flag_selected_col;
        }
        if (bitflag_test(flag, FLAG_SOFTSELECTED, FLAG_SOFTSELECTED)) {
            vertColor = flag_softselected_col;
        }
        //if (!bitflag_isVisible(flag)) {
        //    vertColor = vec4(0.0, 0.0, 0.0, 0.0);
        //}
    }

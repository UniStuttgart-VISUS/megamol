
    // Color  
    vertColor = inColor;
    if (bool(useGlobalCol))  {
        vertColor = globalCol;
    }
    if (bool(useTf)) {
        vertColor = tflookup(inColIdx);
    }
    // Overwrite color depending on flags
    if (flag_visible) {
        if (flag_selected) {
            vertColor = flag_selected_color;
        }
        if (flag_soft_selected) {
            vertColor = flag_softselected_color;
        }
        if (flag_highlighted) {
            vertColor = flag_highlighted_color;
        }
    }


    // Color  
    vertColor = inColor;
    if (bool(useGlobalCol))  {
        vertColor = globalCol;
    }
    if (bool(useTf)) {
        vertColor = tflookup(inColIdx);
    }

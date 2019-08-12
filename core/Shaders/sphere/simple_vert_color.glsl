
    // Color  
    vertColor = inColor;
    if (bool(useGlobalCol))  {
        vertColor = globalCol;
    }
    if (bool(useTf)) {  
        float cid = MAX_COLV - MIN_COLV;    
        if (cid < 0.000001) {
            vertColor = texture(tfTexture, 0.5 / COLTAB_SIZE);
        } else {
            cid = (inColIdx - MIN_COLV) / cid;
            cid = clamp(cid, 0.0, 1.0);
        
            cid *= (1.0 - 1.0 / COLTAB_SIZE);
            cid += 0.5 / COLTAB_SIZE;
        
            vertColor = texture(tfTexture, cid);
        }
    }

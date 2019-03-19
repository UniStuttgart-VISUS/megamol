void main(void) {

    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = inVertex;
    rad = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;
    inPos.w = 1.0;
    
    if (COLTAB_SIZE > 0.0) {  
        float cid = MAX_COLV - MIN_COLV;    
        if (cid < 0.000001) {
            vertColor = texture(colTab, 0.5 / COLTAB_SIZE);
        } else {
            cid = (colIdx - MIN_COLV) / cid;
            cid = clamp(cid, 0.0, 1.0);
        
            cid *= (1.0 - 1.0 / COLTAB_SIZE);
            cid += 0.5 / COLTAB_SIZE;
        
            vertColor = texture(colTab, cid);
        }
    } else {
        vertColor = inColor;
    }
    
#ifdef WITH_SCALING
    rad *= scaling;
#endif // WITH_SCALING
    
    squarRad = rad * rad;
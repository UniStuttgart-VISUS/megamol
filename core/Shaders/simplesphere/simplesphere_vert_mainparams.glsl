void main(void) {

    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = gl_Vertex;
    rad = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;
    inPos.w = 1.0;
    
    if (COLTAB_SIZE > 0.0) {  
    float cid = MAX_COLV - MIN_COLV;    
        if (cid < 0.000001) {
            gl_FrontColor = texture1D(colTab, 0.5 / COLTAB_SIZE);
        } else {
            cid = (colIdx - MIN_COLV) / cid;
            cid = clamp(cid, 0.0, 1.0);
        
            cid *= (1.0 - 1.0 / COLTAB_SIZE);
            cid += 0.5 / COLTAB_SIZE;
        
            gl_FrontColor = texture1D(colTab, cid);
        }
    } else {
        gl_FrontColor = gl_Color;
    }
    
    rad *= scaling;
    
    squarRad = rad * rad;
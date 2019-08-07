void main(void) {
    
    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = inPosition;
    rad = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;
    inPos.w = 1.0;
        
#ifdef WITH_SCALING
    rad *= scaling;
#endif // WITH_SCALING
    
    squarRad = rad * rad;


    vec4 inPos = inPosition;
    inPos.w = 1.0;
    /// DO NOT APPLY (why?)
    /// rad = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;


#ifdef WITH_SCALING
    rad *= scaling;
#endif // WITH_SCALING
    
    squarRad = rad * rad;

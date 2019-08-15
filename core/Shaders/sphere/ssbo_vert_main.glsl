
    vec4 inPos = inPosition;
    inPos.w = 1.0;
    /// DO NOT APPLY (why?):
    /// rad = (constRad < -0.5) ? inPos.w : constRad;

#ifdef WITH_SCALING
    rad *= scaling;
#endif // WITH_SCALING
    
    squarRad = rad * rad;

 #ifdef HALO
    squarRad = (rad + HALO_RAD) * (rad + HALO_RAD);
#endif // HALO

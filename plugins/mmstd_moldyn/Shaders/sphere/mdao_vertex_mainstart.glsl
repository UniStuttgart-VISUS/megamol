
in vec4 inPosition;
in vec4 inColor;
in float inColIdx;

void main(void) {

    // Remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = inPosition;
    rad = (constRad < -0.5) ? inPos.w : constRad;
    inPos.w = 1.0;
        
#ifdef WITH_SCALING
    rad *= scaling;
#endif // WITH_SCALING
    
    squarRad = rad * rad;

    // Position transformations
    
    // object pivot point in object space     
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know) 
 
    // calculate cam position 
    camPos = MVinv[3]; // (C) by Christoph 
    camPos.xyz -= objPos.xyz; // cam pos to glyph space 

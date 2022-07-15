
void main(void) {

    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = gl_Vertex;
    rad = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;
    inPos.w = 1.0;

    gl_FrontColor = gl_Color;

#ifdef WITH_SCALING
    rad *= scaling;
#endif // WITH_SCALING

    squarRad = rad * rad;

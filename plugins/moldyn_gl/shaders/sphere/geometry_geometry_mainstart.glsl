
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 colorgs[1];
in float colidxgs[1];

void main(void) {
    
    // Remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = gl_in[0].gl_Position;
    rad = (constRad < -0.5) ? inPos.w : constRad;
    inPos.w = 1.0;

#ifdef WITH_SCALING
    rad *= scaling;
#endif // WITH_SCALING

    squarRad = rad*rad;
     
#ifdef HALO
    squarRad = (rad + HALO_RAD) * (rad + HALO_RAD);
#endif // HALO

    vec4 inColor = colorgs[0];
    float inColIdx = colidxgs[0];

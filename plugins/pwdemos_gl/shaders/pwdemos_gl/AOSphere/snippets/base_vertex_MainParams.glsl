void main(void) {

    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = gl_Vertex;
    float rad = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;
    inPos.w = 1.0;
    float cid = MAX_COLV - MIN_COLV;
    if (cid < 0.000001) {
        gl_FrontColor = gl_Color;
    } else {
        cid = (colIdx - MIN_COLV) / cid;
        cid = clamp(cid, 0.0, 1.0);

        cid *= (1.0 - 1.0 / COLTAB_SIZE);
        cid += 0.5 / COLTAB_SIZE;

        gl_FrontColor = texture1D(colTab, cid);
    }

#ifdef WITH_SCALING
    rad *= scaling;
#endif // WITH_SCALING
    float radsqr = rad * rad;

    radii = vec2(rad, radsqr);

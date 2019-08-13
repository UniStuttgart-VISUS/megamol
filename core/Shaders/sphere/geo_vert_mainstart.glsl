void main(void) {
    
    gl_Position = inPosition;
    float rad = inPosition.w;
    if (constRad > -0.5) {
      gl_Position.w = constRad;
      rad = constRad;
    }
    
    // clipping
    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        float od = dot(inPosition.xyz, clipDat.xyz) - rad;
        if (od > clipDat.w) {
          gl_Position = vec4(1.0, 1.0, 1.0, 0.0);
        }
    }   

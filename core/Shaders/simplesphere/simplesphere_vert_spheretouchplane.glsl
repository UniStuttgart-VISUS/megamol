    // Sphere-Touch-Plane-Approach
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system
    vec4 tmp;

#ifdef CALC_CAM_SYS
    // camera coordinate system in object space
    tmp = gl_ModelViewMatrixInverse[3] + gl_ModelViewMatrixInverse[2];
    vec3 camIn = normalize(tmp.xyz);
    tmp = gl_ModelViewMatrixInverse[3] + gl_ModelViewMatrixInverse[1];
    vec3 camUp = tmp.xyz;
    vec3 camRight = normalize(cross(camIn, camUp));
    camUp = normalize(cross(camIn, camRight));
#endif // CALC_CAM_SYS

    vec4 mins, maxs;
    vec2 projMins, projMaxs;
    vec3 testPos;
    vec4 projPos;

    // projected camera vector
    vec3 c2 = vec3(dot(camPos.xyz, camRight), dot(camPos.xyz, camUp), dot(camPos.xyz, camIn));

    vec3 cpj1 = camIn * c2.z + camRight * c2.x;
    vec3 cpm1 = camIn * c2.x - camRight * c2.z;

    vec3 cpj2 = camIn * c2.z + camUp * c2.y;
    vec3 cpm2 = camIn * c2.y - camUp * c2.z;
    
    d.x = length(cpj1);
    d.y = length(cpj2);
    
    if (d.x < rad || d.y < rad) {
      od = clipDat.w + 1.0;
    }

    dd = vec2(1.0) / d;

    p = squarRad * dd;
    q = d - p;
    h = sqrt(p * q);
    //h = vec2(0.0);
    
    p *= dd;
    h *= dd;

    cpj1 *= p.x;
    cpm1 *= h.x;
    cpj2 *= p.y;
    cpm2 *= h.y;

    vec4 left, right, top, bottom, front;

    //vec4 oP = gl_ModelViewProjectionMatrix * vec4(objPos.xyz + cpj1, 1.0);
    right = gl_ModelViewProjectionMatrix * vec4(objPos.xyz + cpj1 + cpm1, 1.0);
    top = gl_ModelViewProjectionMatrix * vec4(objPos.xyz + cpj2 + cpm2, 1.0);
    // TODO apparently, this is just wishful thinking. not sure why.
    //left = - (right - oP) + oP;
    //bottom = - (top - oP) + oP;
    left = gl_ModelViewProjectionMatrix * vec4(objPos.xyz + cpj1 - cpm1, 1.0);
    bottom = gl_ModelViewProjectionMatrix * vec4(objPos.xyz + cpj2 - cpm2, 1.0);

    front = gl_ModelViewProjectionMatrix * vec4(objPos.xyz + normalize(camPos.xyz) * rad, 1.0);

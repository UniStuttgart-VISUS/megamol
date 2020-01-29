 
    // Sphere-Touch-Plane-Approach
    
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system
    vec4 tmp;

#ifdef CALC_CAM_SYS
    // camera coordinate system in object space
    tmp = MVinv[3] + MVinv[2];
    vec3 camIn = normalize(tmp.xyz);
    tmp = MVinv[3] + MVinv[1];
    vec3 camUp = tmp.xyz;
    vec3 camRight = normalize(cross(camIn, camUp));
    camUp = cross(camIn, camRight);
#endif // CALC_CAM_SYS

    vec2 mins, maxs;
    vec3 testPos;
    vec4 projPos;

#if 0
    // projected camera vector
    vec3 c2 = vec3(dot(camPos.xyz, camRight), dot(camPos.xyz, camUp), dot(camPos.xyz, camIn));

    vec3 cpj1 = camIn * c2.z + camRight * c2.x;
    vec3 cpm1 = camIn * c2.x - camRight * c2.z;

    vec3 cpj2 = camIn * c2.z + camUp * c2.y;
    vec3 cpm2 = camIn * c2.y - camUp * c2.z;
    
    d.x = length(cpj1);
    d.y = length(cpj2);

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

    // TODO: rewrite only using four projections, additions in homogenous coordinates and delayed perspective divisions.
    testPos = objPos.xyz + cpj1 + cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;

    testPos -= 2.0 * cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz + cpj2 + cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos -= 2.0 * cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);
    
    testPos = objPos.xyz - camIn * rad;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
#else

    vec3 base_vec = (-camPos.xyz);
    float bv = length(base_vec);

    //vec3 cc = vec3(dot(base_vec, camRight), dot(base_vec, camUp), dot(base_vec, camIn));

    //vec3 hv = normalize(-cc.z*camRight+cc.x*camIn);
    //vec3 uv = normalize(cross(hv, base_vec));

    vec3 in_v = normalize(base_vec);
    vec3 up = normalize(cross(in_v, camRight));
    vec3 right = normalize(cross(in_v, up));
    up = normalize(cross(in_v, right));

    float pe = squarRad/bv;
    float h2 = squarRad-pe*pe;
    float ha = sqrt(h2);
    //float qu = bv-pe;
    //float x = ha/qu*(qu-rad+pe);

    vec3 plane_base = objPos.xyz - in_v*pe;
    vec3 corner_0 = plane_base-right*ha-up*ha;
    vec3 corner_1 = plane_base+right*ha-up*ha;
    vec3 corner_2 = plane_base+right*ha+up*ha;
    vec3 corner_3 = plane_base-right*ha+up*ha;

    projPos=MVP*vec4(corner_0,1.0);
    projPos/=projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;

    projPos=MVP*vec4(corner_1,1.0);
    projPos/=projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    projPos=MVP*vec4(corner_2,1.0);
    projPos/=projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    projPos=MVP*vec4(corner_3,1.0);
    projPos/=projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    projPos=MVP*vec4(objPos.xyz - camIn*rad,1.0);
    projPos/=projPos.w;
#endif

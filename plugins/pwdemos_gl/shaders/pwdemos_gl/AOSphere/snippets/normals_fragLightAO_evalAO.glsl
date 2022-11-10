    vec3 aoPos = sphereintersection + objPos.xyz;
    aoPos += normal * aoSampDist;
    aoPos = (aoPos - posOrigin) / posExtents;
    float aoFactor = 1.0 - (texture3D(aoVol, aoPos).r * aoSampFact);

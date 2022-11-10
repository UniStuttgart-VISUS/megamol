void main() {
    vec4 coord;
    vec3 ray;

    vec3 faceNormal = vec3(0.0, 1.0, 0.0);

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform fragment coordinates from view coordinates to object coordinates.
    // OLD coord = gl_ModelViewProjectionMatrixInverse * coord;
    coord = ModelViewProjectionMatrixInverse * coord;
    coord /= coord.w;

    vec3 fragCoordWS = coord.xyz; // fragment coordinates in world-space
    // OLD vec3 camPosWS = gl_ModelViewMatrixInverse[3].xyz; // (C) by Christoph
    vec3 camPosWS = camPosInit.xyz;

    coord -= objPos; // ... and to glyph space
    coord.xyz = ((2.0 * ((dot(quat.xyz, coord.xyz) * quat.xyz) + (quat.w * cross(quat.xyz, coord.xyz)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * coord.xyz));

    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);

    // OLD vec4 col = gl_Color;
    vec4 col = color;

    float lambda = 1000000.0;
    vec3 normal;

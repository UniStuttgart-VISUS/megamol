
uniform vec4 viewAttr;
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;

varying vec4 quat;
varying vec4 camPos;
varying vec4 objPos;
varying vec4 lightPos;
varying float rad;

void main() {
    quat = gl_MultiTexCoord0 * vec4(-1.0, -1.0, -1.0, 1.0); // inverted/conjugated quaternion
    vec4 inPos = gl_Vertex;
    rad = inPos.w;
    float squarRad = rad * rad * OUTERRAD * OUTERRAD;
    inPos.w = 1.0;

    // object pivot point in object space    
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)

    // calculate cam position
    camPos = gl_ModelViewMatrixInverse[3]; // (C) by Christoph
    camPos.xyz -= objPos.xyz; // cam pos to glyph space
    // rotate camera AFTER silhouette approximation

    // calculate light position in glyph space
    lightPos = gl_ModelViewMatrixInverse * gl_LightSource[0].position;
    // Don't rotate! Otherwise we would need to rotate the half-way-vector too

    // Sphere-Touch-Plane-Approach™
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system
    vec4 tmp;

    vec2 mins, maxs;
    vec3 testPos;
    vec4 projPos;

    // projected camera vector
    vec3 c2 = vec3(dot(camPos.xyz, camRight), dot(camPos.xyz, camUp), dot(camPos.xyz, camIn));

    // rotate camera AFTER silhouette approximation
    camPos.xyz = ((2.0 * ((dot(quat.xyz, camPos.xyz) * quat.xyz) + (quat.w * cross(quat.xyz, camPos.xyz)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * camPos.xyz));

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

    p *= dd;
    h *= dd;

    cpj1 *= p.x;
    cpm1 *= h.x;
    cpj2 *= p.y;
    cpm2 *= h.y;

    testPos = objPos.xyz + cpj1 + cpm1;
    projPos = gl_ModelViewProjectionMatrix * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;

    testPos -= 2.0 * cpm1;
    projPos = gl_ModelViewProjectionMatrix * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz + cpj2 + cpm2;
    projPos = gl_ModelViewProjectionMatrix * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos -= 2.0 * cpm2;
    projPos = gl_ModelViewProjectionMatrix * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    gl_Position = vec4((mins + maxs) * 0.5, 0.0, 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    gl_PointSize = max(maxs.x, maxs.y) + 0.5;
    gl_FrontColor = gl_Color;
}

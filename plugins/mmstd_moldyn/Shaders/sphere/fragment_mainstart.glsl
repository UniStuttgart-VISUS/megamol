
void main(void) {

    vec4 coord;
    vec3 ray;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    
    // transform fragment coordinates from view coordinates to object coordinates.
    //coord = MVPinv * coord;
    coord = MVPinv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space

    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);

    // chose color for lighting
    vec4 color = vertColor;
    //vec4 color = vec4(uplParams.xyz, 1.0);

    // calculate the geometry-ray-intersection
    float b = -dot(camPos.xyz, ray);           // projected length of the cam-sphere-vector onto the ray
    vec3 temp = camPos.xyz + b*ray;
    float delta = squarRad - dot(temp, temp);  // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)

#ifdef CLIP
    if (delta < 0.0) { 
#ifdef DISCARD_COLOR_MARKER
        color = vec4(1.0, 0.0, 0.0, 1.0);       
#else // DISCARD_COLOR_MARKER
        discard; 
#endif // DISCARD_COLOR_MARKER
    }
#endif // CLIP

    float c = dot(camPos.xyz, camPos.xyz)-squarRad;

    float s = b < 0.0f ? -1.0f : 1.0f;
    float q = b + s*sqrt(delta);
    lambda = min(c/q, q);

    vec3 sphereintersection = lambda * ray + camPos.xyz;    // intersection point
    vec3 normal = sphereintersection / rad;

    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        vec3 planeNormal = normalize(clipDat.xyz);
        vec3 clipPlaneBase = planeNormal * clipDat.w;
        float d = -dot(planeNormal, clipPlaneBase - objPos.xyz);
        float dist1 = dot(sphereintersection, planeNormal) + d;
        float dist2 = d;
        float t = -(dot(planeNormal, camPos.xyz) + d) / dot(planeNormal, ray);
        vec3 planeintersect = camPos.xyz + t * ray;
        if (dist1 > 0.0) {
            if (dist2 < rad) {
                if (length(planeintersect) < rad) {
                    sphereintersection = planeintersect;
                    normal = planeNormal;
                    color = mix(color, vec4(clipCol.rgb, 1.0), clipCol.a);
                } else {
                    discard;
                }
            } else {
                discard;
            }
        }
    }

    // "calc" normal at intersection point
#ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, outlightDir.w);
#endif // SMALL_SPRITE_LIGHTING

#ifdef AXISHINTS
    // debug-axis-hints
    float mc = min(abs(normal.x), min(abs(normal.y), abs(normal.z)));
    if (mc < 0.05)            { color = vec3(0.5); }
    if (abs(normal.x) > 0.98) { color = vec3(1.0, 0.0, 0.0); }
    if (abs(normal.y) > 0.98) { color = vec3(0.0, 1.0, 0.0); }
    if (abs(normal.z) > 0.98) { color = vec3(0.0, 0.0, 1.0); }
    if (normal.x < -0.99)     { color = vec3(0.5); }
    if (normal.y < -0.99)     { color = vec3(0.5); }
    if (normal.z < -0.99)     { color = vec3(0.5); }
#endif // AXISHINTS


vec3 computeRay(vec4 fragCoord, vec4 viewAttr, mat4 MVPinv, vec4 camPos, vec4 objPos){
    // transform fragment coordinates from window coordinates to view coordinates.
    vec4 coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    
    // transform fragment coordinates from view coordinates to object coordinates.
    //coord = MVPinv * coord;
    coord = MVPinv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space

    // calc the viewing ray
    return normalize(coord.xyz - camPos.xyz);
};

struct Intersection{
    vec3 position;
    vec3 normal;
    vec4 color;
};

Intersection computeRaySphereIntersection(vec3 ray, vec4 camPos, vec4 objPos, vec4 color, float squarRad, float rad, vec4 clipDat, vec4 clipCol){
    Intersection retval;
    retval.color = color;

    // calculate the geometry-ray-intersection
    float b = -dot(camPos.xyz, ray);           // projected length of the cam-sphere-vector onto the ray
    vec3 temp = camPos.xyz + b*ray;
    float delta = squarRad - dot(temp, temp);  // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)

#ifdef CLIP
    if (delta < 0.0) {
#ifdef DISCARD_COLOR_MARKER
        retval.color = vec4(1.0, 0.0, 0.0, 1.0);       
#else // DISCARD_COLOR_MARKER
        discard; 
#endif // DISCARD_COLOR_MARKER
    }
#endif // CLIP

    float c = dot(camPos.xyz, camPos.xyz)-squarRad;

    float s = b < 0.0f ? -1.0f : 1.0f;
    float q = b + s*sqrt(delta);
    float lambda = min(c/q, q);

    retval.position = lambda * ray + camPos.xyz;    // intersection point
    retval.normal = retval.position / rad;

    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        vec3 planeNormal = normalize(clipDat.xyz);
        vec3 clipPlaneBase = planeNormal * clipDat.w;
        float d = -dot(planeNormal, clipPlaneBase - objPos.xyz);
        float dist1 = dot(retval.position, planeNormal) + d;
        float dist2 = d;
        float t = -(dot(planeNormal, camPos.xyz) + d) / dot(planeNormal, ray);
        vec3 planeintersect = camPos.xyz + t * ray;
        if (dist1 > 0.0) {
            if (dist2 < rad) {
                if (length(planeintersect) < rad) {
                    retval.position = planeintersect;
                    retval.normal = planeNormal;
                    retval.color = mix(retval.color, vec4(clipCol.rgb, 1.0), clipCol.a);
                } else {
                    discard;
                }
            } else {
                discard;
            }
        }
    }

    return retval;
};

vec4 axisHintsColor(vec3 normal){
    // debug-axis-hints
    vec4 retval = vec4(0.0);
    float mc = min(abs(normal.x), min(abs(normal.y), abs(normal.z)));
    if (mc < 0.05) {
        retval = vec4(0.5, 0.5, 0.5, 1.0);
    }
    if (abs(normal.x) > 0.98) {
        retval = vec4(1.0, 0.0, 0.0, 1.0);
    }
    if (abs(normal.y) > 0.98) {
        retval = vec4(0.0, 1.0, 0.0, 1.0);
    }
    if (abs(normal.z) > 0.98) {
        retval = vec4(0.0, 0.0, 1.0, 1.0);
    }
    if (normal.x < -0.99) {
        retval = vec4(0.5, 0.5, 0.5, 1.0);
    }
    if (normal.y < -0.99) {
        retval = vec4(0.5, 0.5, 0.5, 1.0);
    }
    if (normal.z < -0.99) {
        retval = vec4(0.5, 0.5, 0.5, 1.0);
    }
    return retval;
};

float computeDepthValue(vec3 position, mat4 MVPtransp){
    float depth = dot(MVPtransp[2], vec4(position, 1.0));
    float depthW = dot(MVPtransp[3], vec4(position, 1.0));
    return ((depth / depthW) + 1.0) * 0.5;
};

vec4 reticleColor(vec4 color, vec4 fragCoord, vec2 sphere_frag_center){
    vec4 retval = color;
    if (min(abs(fragCoord.x - sphere_frag_center.x), abs(fragCoord.y - sphere_frag_center.y)) < 2.0f) {
        //outColor.rgb = vec3(1.0, 1.0, 0.5);
        retval.rgb += vec3(0.3, 0.3, 0.5);
    }
    return retval;
};

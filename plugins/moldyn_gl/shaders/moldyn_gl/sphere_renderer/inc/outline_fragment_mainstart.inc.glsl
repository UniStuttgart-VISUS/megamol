
void main(void) {

    vec4 coord;
    vec3 ray;
    float lambda;

    vec4 color = vertColor;
    float distance = length(gl_FragCoord.xy - sphere_frag_center);

#ifdef CLIP

    if (distance > (sphere_frag_radius + outlineWidth)) {
#ifdef DISCARD_COLOR_MARKER
        color = vec4(1.0, 0.0, 0.0, 1.0);
#else // DISCARD_COLOR_MARKER
        discard;
#endif // DISCARD_COLOR_MARKER
    }

#endif // CLIP

    if (length(gl_FragCoord.xy - sphere_frag_center) < sphere_frag_radius) {
        discard;
    }
    outColor = color;

// Calculate depth
#ifdef DEPTH

    vec4 Ding = vec4(objPos.xyz, 1.0);
    float depth = dot(MVPtransp[2], Ding);
    float depthW = dot(MVPtransp[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;

#endif // DEPTH

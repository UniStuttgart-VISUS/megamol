#undef HALO
//#define HALO
#undef HALO_RAD
#define HALO_RAD  0.1

uniform vec4 viewAttr;

varying vec4 objPos;
varying vec4 camPos;
varying vec4 lightPos;
varying float squarRad;
varying float rad;

#ifdef RETICLE
varying vec2 centerFragment;
#endif // RETICLE

void main(void) {
    vec4 coord;
    vec3 ray;
    float lambda;
    vec3 color;
    vec3 sphereintersection = vec3( 0.0);
    vec3 normal;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);


    // transform fragment coordinates from view coordinates to object coordinates.
    coord = gl_ModelViewProjectionMatrixInverse * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space


    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);


    // calculate the geometry-ray-intersection
    float d1 = -dot(camPos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(camPos.xyz, camPos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = squarRad - d2s;                        // square of difference of projected length and lambda
#ifdef CLIP
    lambda = d1 - sqrt(radicand);                           // lambda

    float radicand2 = 0.0;
#ifdef HALO
    radicand2 = (rad+HALO_RAD)*(rad+HALO_RAD) - d2s;
    if( radicand2 < 0.0 ) {
        discard;
    }
    else if( radicand < 0.0 ) {
        // idea for halo from Tarini et al. (tvcg 2006)
        color = vec3(0.1);
        normal = vec3(0.0, 1.0, 0.0);
    }
#else
    if( radicand < 0.0 ) {
        discard;
    }
#endif // HALO
    else {
        // chose color for lighting
        color = gl_Color.rgb;
        sphereintersection = lambda * ray + camPos.xyz;    // intersection point
        // "calc" normal at intersection point
        normal = sphereintersection / rad;
    }

#endif // CLIP

#ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, lightPos.w);
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

    // phong lighting with directional light
    //gl_FragColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    gl_FragColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color), gl_Color.w);
    gl_FragDepth = gl_FragCoord.z;

    // calculate depth
#ifdef DEPTH
    vec4 Ding = vec4( sphereintersection + objPos.xyz, 1.0);
    float depth = dot(gl_ModelViewProjectionMatrixTranspose[2], Ding);
    float depthW = dot(gl_ModelViewProjectionMatrixTranspose[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;

#ifndef CLIP
    gl_FragDepth = (radicand < 0.0) ? 1.0 : ((depth / depthW) + 1.0) * 0.5;
    gl_FragColor.rgb = (radicand < 0.0) ? gl_Color.rgb : gl_FragColor.rgb;
#endif // CLIP

#endif // DEPTH


#ifdef RETICLE
    coord = gl_FragCoord
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);
    if (min(abs(coord.x - centerFragment.x), abs(coord.y - centerFragment.y)) < 0.002) {
        //gl_FragColor.rgb = vec3(1.0, 1.0, 0.5);
        gl_FragColor.rgb += vec3(0.3, 0.3, 0.5);
    }
#endif // RETICLE

#ifdef PUXELS
if(puxels_use != 0)
    puxels_store(makePuxel(packUnorm4x8(gl_FragColor), normal, gl_FragDepth));
#endif

    //gl_FragColor.rgb = normal;
    //gl_FragColor.rgb = lightPos.xyz;
}

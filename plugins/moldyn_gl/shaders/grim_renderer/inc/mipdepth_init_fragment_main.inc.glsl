uniform vec4 viewAttr;
// new camera
uniform mat4 mv_inv;
uniform mat4 mv_inv_transp;
uniform mat4 mvp;
uniform mat4 mvp_inv;
uniform mat4 mvp_transp;
uniform vec4 light_dir;
uniform vec4 cam_pos;
// end new camera
FLACH varying vec4 objPos;
FLACH varying vec4 camPos;
FLACH varying float squarRad;
FLACH varying float rad;

void main(void) {
    vec4 coord;
    vec3 ray;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform fragment coordinates from view coordinates to object coordinates.
    coord = mvp_inv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space

    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);

    // calculate the geometry-ray-intersection
    float d1 = -dot(camPos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(camPos.xyz, camPos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = squarRad - d2s;                        // square of difference of projected length and lambda
    if (radicand < 0.0) {
        discard;
    }

    gl_FragColor = gl_Color;

}

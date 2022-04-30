float rand(vec2 co) {
    return fract(sin(dot(co.xy ,vec2(12.9898, 78.233))) * 43758.5453);
}

void main(void) {

// some bullshit lighting to fake the impression of smart things happening
    float fac = 0.5 + 0.45 * rand(gl_FragCoord.xy + gl_FragCoord.zz);
    float light = max(0.0, rand(gl_FragCoord.yx + gl_FragCoord.zz) * 9.5 - 9.0);
    gl_FragColor = vec4(gl_Color.rgb * fac + vec3(1.0, 1.0, 1.0) * light, gl_Color.a);

}

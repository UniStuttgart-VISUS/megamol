uniform sampler2D datex;
uniform ivec2 dst;
uniform ivec2 src;

void main(void) {
    ivec2 c = 2 * (ivec2(gl_FragCoord.xy) - dst.xy) + src.xy;
    vec4 c1 = texelFetch2D(datex, c, 0);
    vec4 c2 = texelFetch2D(datex, c + ivec2(1, 0), 0);
    vec4 c3 = texelFetch2D(datex, c + ivec2(0, 1), 0);
    vec4 c4 = texelFetch2D(datex, c + ivec2(1, 1), 0);

    gl_FragColor = vec4(max(max(c1.r, c2.r), max(c3.r, c4.r)));
}

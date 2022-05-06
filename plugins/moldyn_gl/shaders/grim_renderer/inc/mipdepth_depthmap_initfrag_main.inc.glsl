
uniform sampler2D datex;

void main(void) {
    gl_FragColor = texelFetch2D(datex, ivec2(gl_FragCoord.xy), 0);
}

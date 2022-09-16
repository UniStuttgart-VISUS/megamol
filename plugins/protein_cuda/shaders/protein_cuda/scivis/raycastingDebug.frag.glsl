varying vec3 posES;
void main (void) {
    gl_FragData[0] = vec4(gl_TexCoord[0].stp, 1.0f);
    gl_FragData[1] = vec4(posES, 1.0f);
    gl_FragData[2] = vec4(gl_FragCoord.xyz, 1.0);
}

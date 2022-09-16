varying vec3 posES;
void main(void) {
    posES = (gl_ModelViewMatrix*gl_Vertex).xyz;
    gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex;
    gl_TexCoord[0] = gl_MultiTexCoord0;

}

#version 130

void main(void) {
    gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex;
    gl_TexCoord[0] = gl_MultiTexCoord0; // 3D tex coords
    gl_TexCoord[1] = gl_MultiTexCoord1; // 2D tex coords
}

uniform sampler2D matrixTex;
uniform float minVal;
uniform float maxVal;

void main() {
    float val = clamp(texture2D(matrixTex, gl_TexCoord[0].st).a, minVal, maxVal);
    //vec3 colOrangeMsh = vec3(109.81, 0.9746*(val-minVal)/(maxVal-minVal), 0.8968);
    vec3 colYellowMsh = vec3(102.44, 0.6965*(val-minVal)/(maxVal-minVal), 1.5393);
    gl_FragColor = vec4(MSH2RGB(colYellowMsh.r, colYellowMsh.g, colYellowMsh.b), 1.0);
}

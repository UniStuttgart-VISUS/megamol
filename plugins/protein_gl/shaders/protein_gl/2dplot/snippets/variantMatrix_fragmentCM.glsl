uniform float minVal;
uniform float maxVal;
varying float valFrag;

void main() {
    //vec3 colOrangeMsh = vec3(109.81, 0.9746*(valFrag-minVal)/(maxVal-minVal), 0.8968);
    vec3 colYellowMsh = vec3(102.44, 0.6965*(valFrag-minVal)/(maxVal-minVal), 1.5393);
    gl_FragColor = vec4(MSH2RGB(colYellowMsh.r, colYellowMsh.g, colYellowMsh.b), 1.0);
    //gl_FragColor = vec4(valFrag,0.0,0.0, 1.0);
}

in float binColor;
in float selection;

void main(void) {
    if (selection <= 1.0) {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        gl_FragColor = tflookup(binColor);
    }
}

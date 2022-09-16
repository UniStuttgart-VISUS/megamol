uniform vec4 color;
void main() {
  // OLD gl_FragColor = vec4(gl_Color.xyz, 1.0); // whoho fancy
  gl_FragColor = vec4(color.xyz, 1.0); // whoho fancy
}

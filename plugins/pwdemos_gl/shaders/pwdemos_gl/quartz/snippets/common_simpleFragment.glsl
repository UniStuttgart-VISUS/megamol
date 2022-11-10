uniform vec4 color;

layout(location = 0) out vec4 outColor;

void main() {
  // OLD gl_FragColor = vec4(gl_Color.xyz, 1.0); // whoho fancy
  outColor = vec4(color.xyz, 1.0); // whoho fancy
}

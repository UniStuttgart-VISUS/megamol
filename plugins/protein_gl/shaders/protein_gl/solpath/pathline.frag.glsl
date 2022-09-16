varying vec2 values;

void main() {
  float t = values.x;
  t *= 3.0;
  vec3 c, cd;
  if (t < 1.0) {
    c = vec3(1.0, 0.0, 0.0);
    cd = vec3(0.0, 1.0, 0.0);
  } else if (t < 2.0) {
    c = vec3(1.0, 1.0, 0.0);
    cd = vec3(-1.0, 0.0, 1.0);
    t -= 1.0;
  } else {
    c = vec3(0.0, 1.0, 1.0);
    cd = vec3(0.0, -1.0, 0.0);
    t -= 2.0;
  }

  float a = values.y;
  float b = 1.0 - a;
  gl_FragColor = vec4((c + cd * t) * a + b * vec3(0.3, 0.3, 0.3), 1.0);
}

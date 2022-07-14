
#extension GL_EXT_gpu_shader4 : enable

varying vec3 opos;

void main(void) {
  gl_FragData[0] = vec4(gl_Color.rgb, 1.0);
  gl_FragData[1] = vec4(0.0, 0.0, 0.0, 0.0);
  gl_FragData[2] = vec4(opos, 1.0);
}

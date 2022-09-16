// see https://github.com/bernstein/pixelparty-shader/blob/master/hsv.glsl
vec3 HSV2RGB(vec3 hsv) {

  vec3 rgb = vec3(hsv.z);
  if ( hsv.y != 0.0 ) {
    float var_h = hsv.x * 6.0;
    float var_i = floor(var_h);
    float var_1 = hsv.z * (1.0 - hsv.y);
    float var_2 = hsv.z * (1.0 - hsv.y * (var_h-var_i));
    float var_3 = hsv.z * (1.0 - hsv.y * (1.0 - (var_h-var_i)));

    switch (int(var_i)) {
      case 0: rgb = vec3(hsv.z, var_3, var_1); break;
      case 1: rgb = vec3(var_2, hsv.z, var_1); break;
      case 2: rgb = vec3(var_1, hsv.z, var_3); break;
      case 3: rgb = vec3(var_1, var_2, hsv.z); break;
      case 4: rgb = vec3(var_3, var_1, hsv.z); break;
      default: rgb = vec3(hsv.z, var_1, var_2); break;
    }
  }
  return rgb;
}

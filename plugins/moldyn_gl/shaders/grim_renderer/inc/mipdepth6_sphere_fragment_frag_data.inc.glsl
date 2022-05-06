
if (bool(use_shading)) {
    gl_FragData[0] = vec4(LocalLighting(ray, normal, lightDir.xyz, color.rgb), 1.0);
  } else {
    gl_FragData[0] = vec4(color.rgb, 1.0);
  }

  //normal = normalize((mv_inv_transp * vec4(normal, 1.0)).xyz);
  gl_FragData[1] = vec4(normal.xyz, clamp((pointSize - 4.0) * 0.25, 0.0, 1.0));
  gl_FragData[2] = vec4(sphereintersection.xyz + objPos.xyz, 1.0);
}

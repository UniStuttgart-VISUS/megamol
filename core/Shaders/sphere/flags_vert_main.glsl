 #ifdef FLAG_STORAGE

  if ((flags[(flags_offset + gl_VertexID)] & FLAG_SELECTED) == FLAG_SELECTED) {
    vertColor = vec4(0.0, 0.0, 0.0, 1.0);
  } 
  //else {
  //    vertColor = ...;
  //}

#endif // FLAG_STORAGE
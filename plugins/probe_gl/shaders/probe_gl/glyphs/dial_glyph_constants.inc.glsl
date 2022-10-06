const vec4 glyph_border_color = vec4(0.0,0.0,0.0,1.0);

const float base_line_width = 0.02;
const float inner_radius = 0.333;
const float arrow_base_radius = 0.2;
const float angle_start = 0.05;
const float angle_end = 0.95;
const float angle_arrow_start = 0.85;

const vec4 vertices[6] = vec4[6]( vec4( -1.0,-1.0,0.0,0.0 ),
                                      vec4( 1.0,1.0,1.0,1.0 ),
                                      vec4( -1.0,1.0,0.0,1.0 ),
                                      vec4( 1.0,1.0,1.0,1.0 ),
                                      vec4( -1.0,-1.0,0.0,0.0 ),
                                      vec4( 1.0,-1.0,1.0,0.0 ) );

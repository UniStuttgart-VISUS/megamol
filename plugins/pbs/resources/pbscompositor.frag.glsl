#version 130

//layout(location=0) out vec4 outCol;

uniform sampler2D color;

uniform sampler2D depth;

in vec3 ws_pos;

void main() {
    vec2 coord = vec2((ws_pos.x+1)/2, (ws_pos.y+1)/2);

    gl_FragDepth = texture(depth, coord).r;

    gl_FragColor = texture(color, coord);
    //outCol = texture(color, coord);
    //outCol = vec4(1.0);
}
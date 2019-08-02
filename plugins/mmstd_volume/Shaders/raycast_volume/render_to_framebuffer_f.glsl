uniform sampler2D src_tx2D;
uniform sampler2D depth_tx2D;

in vec2 uv_coord;

out vec4 frag_out;

void main()
{
    frag_out = texture(src_tx2D, uv_coord);
    gl_FragDepth = texture(depth_tx2D, uv_coord).x;
}
uniform sampler2D src_tx2D;
uniform sampler2D normal_tx2D;
uniform sampler2D depth_tx2D;

in vec2 uv_coord;

layout (location = 0) out vec4 frag_out;
layout (location = 1) out vec4 normal_out;

void main()
{
    frag_out = texture(src_tx2D, uv_coord);
    normal_out = texture(normal_tx2D, uv_coord);

    gl_FragDepth = texture(depth_tx2D, uv_coord).x;
}

#version 430

uniform sampler2D src_tx2D;
uniform sampler2D normal_tx2D;
uniform sampler2D depth_tx2D;
uniform sampler1D tf_tx1D;

uniform vec2 valRange;

in vec2 uv_coord;

layout (location = 0) out vec4 frag_out;
layout (location = 1) out vec4 normal_out;

void main()
{
    frag_out = texture(src_tx2D, uv_coord);
    vec4 vol_sample = texture(tf_tx1D, (frag_out.w - valRange.x) / (valRange.y - valRange.x));
    frag_out = vec4(frag_out.xyz * (1.0f - vol_sample.w) + vol_sample.xyz * vol_sample.w, 1.0f);

    normal_out = texture(normal_tx2D, uv_coord);

    gl_FragDepth = texture(depth_tx2D, uv_coord).x;
}

uniform sampler2D input_tx2D;
uniform sampler2D depth_tx2D;

uniform bool flags_available = false;
uniform uint frame_id = 0;
uniform ivec2 viewport_resolution = ivec2(0, 0);
layout(std430, binding = 5) buffer Flags
{
    coherent uint flags[];
};

in vec2 uv_coord;

out layout(location = 0) vec4 frag_out;


void main(void) {
    if (flags_available) {
        ivec2 texSize = textureSize(input_tx2D, 0).xy;
        ivec2 texPos = ivec2((gl_FragCoord.xy / viewport_resolution) * texSize);
        uint rowId = uint(texPos.y) * texSize.x + uint(texPos.x);
        if (bitflag_isVisible(flags[rowId])) {
            if (bitflag_isVisibleSelected(flags[rowId])) {
                vec2 uv = (gl_FragCoord.xy - .5*texSize.xy) / texSize.y;
                float zebra = abs(4. *fract((uv.x + uv.y + float(frame_id)*0.0005) * 60.)-1.);
                zebra = clamp(zebra, 0.0, 1.0);
                vec4 col = texture(input_tx2D,uv_coord).rgba;
                frag_out = vec4(col.rgb * zebra + (1 - col.rgb) * (1 - zebra), col.a);
            } else {
                frag_out = texture(input_tx2D,uv_coord).rgba;
            }
        }
    } else {
        frag_out = texture(input_tx2D,uv_coord).rgba;
    }

    gl_FragDepth = texture(depth_tx2D, uv_coord).r;
}

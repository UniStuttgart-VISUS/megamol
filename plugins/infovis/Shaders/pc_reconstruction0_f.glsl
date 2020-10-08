uniform sampler2DMSArray src_tex2D;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

uniform mat4 moveMatrices[2];

in vec2 uv_coord;
out vec4 frag_out;

void main()
{
        vec4 p = vec4(2 * uv_coord.x - 1, 2 * uv_coord.y - 1, 1, 1);
        int col = int(uv_coord.x * w) % 2;
        int lin = int(uv_coord.y * h) % 2;
        int ind = int(col + lin) % 2;
        p = moveMatrices[ind] * p + vec4(1,1,0,0);
        p.x = 0.5 * w * p.x;
        p.y = 0.5 * h * p.y;
        frag_out = texelFetch(src_tex2D, ivec3( (p.x / 2) - int(lin == 1 && col == 0), p.y / 2 , ind), lin + 1 % 2);
}
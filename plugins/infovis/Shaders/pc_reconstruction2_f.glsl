uniform sampler2D src_tx2Da;
uniform sampler2D src_tx2Db;
uniform sampler2D src_tx2Dc;
uniform sampler2D src_tx2Dd;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

uniform mat4 mMa;
uniform mat4 mMb;
uniform mat4 mMc;
uniform mat4 mMd;

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
	vec4 p = vec4(2 * uv_coord.x - 1, 2 * uv_coord.y - 1, 1, 1);
	frag_out = 0.25 * texture(src_tx2Dd, 0.5 * (mMd * p).xy + vec2(0.5));
	frag_out += 0.25 * texture(src_tx2Dc, 0.5 * (mMc * p).xy + vec2(0.5));
	frag_out += 0.25 * texture(src_tx2Da,  0.5 * (mMa * p).xy + vec2(0.5));
	frag_out += 0.25 * texture(src_tx2Db, 0.5 * (mMb * p).xy + vec2(0.5));

}
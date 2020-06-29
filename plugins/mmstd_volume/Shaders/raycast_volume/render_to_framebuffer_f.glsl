uniform sampler2D src_tx2D;
uniform sampler2D src_tx2Db;

uniform int frametype;

in vec2 uv_coord;

out vec4 frag_out;

void main()
{
    if(frametype == 0){
        frag_out = texture(src_tx2D, uv_coord);
    }
    if(frametype == 1) {
        frag_out = texture(src_tx2D, uv_coord);
    }
    if(frametype == 2) {
        frag_out = texture(src_tx2Db, uv_coord);
    }
}
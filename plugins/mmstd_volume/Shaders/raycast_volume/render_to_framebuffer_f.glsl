uniform sampler2D src_tx2D;
uniform sampler2D src_tx2Db;

uniform int frametype;
uniform int h;

in vec2 uv_coord;

out vec4 frag_out;

void main()
{
    if(frametype == 0){
        frag_out = texture(src_tx2D, uv_coord);
    }else{
        if(int(uv_coord.y*h) % 2 == 0){
            frag_out = texture(src_tx2D, uv_coord);
            //frag_out = vec4(1,0,0,1);
        }
        if(int(uv_coord.y*h) % 2 == 1){
            //frag_out = vec4(0,0,1,1);
            frag_out = texture(src_tx2Db, vec2(uv_coord.x, uv_coord.y+(1.0/h)));
        }
    }
}
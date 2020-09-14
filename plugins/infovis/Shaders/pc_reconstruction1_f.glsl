uniform sampler2D src_tx2Da;
uniform sampler2D src_tx2Db;
uniform sampler2D src_tx2Dc;
uniform sampler2D src_tx2Dd;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
    if(int(gl_FragCoord.y) % 2 == 1){
        if(int(gl_FragCoord.x) % 2 == 1){
            frag_out = texture(src_tx2Dd, vec2(uv_coord.x + (1.0/w), uv_coord.y - (1.0/h)));
            //frag_out = vec4(1.0, 0.0, 0.0, 1.0);
        }else{
            frag_out = texture(src_tx2Dc, vec2(uv_coord.x, uv_coord.y - (1.0/h)));
            //frag_out = vec4(0.0, 1.0, 0.0, 1.0);
        }
    } else {
        if(int(gl_FragCoord.x) % 2 == 0){
            frag_out = texture(src_tx2Da, vec2(uv_coord.x, uv_coord.y));
            //frag_out = vec4(0.0, 0.0, 1.0, 1.0);
        }else{
            frag_out = texture(src_tx2Db, vec2(uv_coord.x + (1.0/w), uv_coord.y));
            //frag_out = vec4(1.0, 1.0, 1.0, 1.0);
        }  
    }
    //frag_out = vec4(uv_coord, 1.0 , 1.0);
    //frag_out = vec4(gl_FragCoord.x / w, gl_FragCoord.y / h, 0.0 , 1.0);  
}
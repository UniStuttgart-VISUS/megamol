uniform sampler2DMS src_tx2D;
uniform sampler2DMS src_tx2Db;

uniform int frametype;
uniform int h;
uniform int w;

in vec2 uv_coord;
layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
    if(true){
        if(int(uv_coord.y * h) % 2 == 1){
            if(int(uv_coord.x * w) % 2 == 0){
                frag_out = texelFetch(src_tx2D, ivec2(uv_coord.x * w/2, uv_coord.y * h/2), 0);
            }else{
                frag_out = texelFetch(src_tx2Db, ivec2(uv_coord.x * w/2, uv_coord.y * h/2), 0);
            }
        } else {
          if(int(uv_coord.x * w) % 2 == 1){
                frag_out = texelFetch(src_tx2D, ivec2(uv_coord.x * w/2, uv_coord.y * h/2), 1);
            }else{
                frag_out = texelFetch(src_tx2Db, ivec2(uv_coord.x/2 * w, uv_coord.y * h/2), 1);
            }  
        }
    }
}
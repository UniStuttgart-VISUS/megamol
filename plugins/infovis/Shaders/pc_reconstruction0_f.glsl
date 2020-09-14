uniform sampler2DMS src_tx2Da;
uniform sampler2DMS src_tx2Db;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

in vec2 uv_coord;
layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
    if(true){
        if(int(gl_FragCoord.y) % 2 == 1){
            if(int(gl_FragCoord.x) % 2 == 1){
                frag_out = texelFetch(src_tx2Da, ivec2(gl_FragCoord.x / 2, gl_FragCoord.y / 2), 0);
                //frag_out = vec4(1.0, 0.0, 0.0, 1.0);
            }else{
                frag_out = texelFetch(src_tx2Db, ivec2((gl_FragCoord.x -2) / 2, gl_FragCoord.y / 2), 0);
                //frag_out = vec4(0.0, 1.0, 0.0, 1.0);
            }
        } else {
          if(int(gl_FragCoord.x) % 2 == 0){
                frag_out = texelFetch(src_tx2Da, ivec2(gl_FragCoord.x / 2, gl_FragCoord.y / 2), 1);
                //frag_out = vec4(0.0, 0.0, 1.0, 1.0);
            }else{
                frag_out = texelFetch(src_tx2Db, ivec2((gl_FragCoord.x) / 2, gl_FragCoord.y / 2), 1);
                //frag_out = vec4(1.0, 1.0, 1.0, 1.0);
            }  
        }
        //frag_out = vec4(uv_coord, 1.0 , 1.0);
        //frag_out = vec4(gl_FragCoord.x / w, gl_FragCoord.y / h, 0.0 , 1.0);
    }
    if(frametype == 3){
        frag_out = vec4(float(int(uv_coord.x * w) % 2), float(int(uv_coord.y * h) % 2), 1.0 , 1.0);
    }

}
uniform sampler2DMS src_tx2Da;
uniform sampler2DMS src_tx2Db;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

uniform mat4 mMa;
uniform mat4 mMb;

in vec2 uv_coord;
layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
        
        if(int(gl_FragCoord.y) % 2 == 1){
            if(int(gl_FragCoord.x) % 2 == 1){
                vec4 p = mMa * vec4(2 * uv_coord.x, 2 * uv_coord.y, 1, 1);
                p.x = p.x * (0.5*w);
                p.y = p.y * (0.5*h);
                frag_out = texelFetch(src_tx2Da, ivec2(int(p.x) / 2, p.y / 2), 0);
            }else{
                vec4 p = mMb * vec4(2 * uv_coord.x, 2 * uv_coord.y, 1, 1);
                p.x = p.x * (0.5*w);
                p.y = p.y * (0.5*h);
                frag_out = texelFetch(src_tx2Db, ivec2((int(p.x) -2) / 2, p.y / 2), 0);
            }
        } else {
          if(int(gl_FragCoord.x) % 2 == 0){
                vec4 p = mMa * vec4(2 * uv_coord.x, 2 * uv_coord.y, 1, 1);
                p.x = p.x * (0.5*w);
                p.y = p.y * (0.5*h);
                frag_out = texelFetch(src_tx2Da, ivec2(int(p.x) / 2, p.y / 2), 1);
            }else{
                vec4 p = mMb * vec4(2 * uv_coord.x, 2 * uv_coord.y, 1, 1);
                p.x = p.x * (0.5*w);
                p.y = p.y * (0.5*h);
                frag_out = texelFetch(src_tx2Db, ivec2(int(p.x) / 2, p.y / 2), 1);
            }  
        }

}
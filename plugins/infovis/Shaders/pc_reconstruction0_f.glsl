uniform sampler2DMS samplers[2];
uniform mat4 mMatrices[2];

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

in vec2 uv_coord;
layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
        vec4 p = vec4(2 * uv_coord.x - 1, 2 * uv_coord.y - 1, 1, 1);
        if(int(gl_FragCoord.y) % 2 == 1){
            if(int(gl_FragCoord.x) % 2 == 1){
                p = mMatrices[0] * p + vec4(1, 1, 0, 0);
                p.x = p.x * (0.5*w);
                p.y = p.y * (0.5*h);
                frag_out = texelFetch(samplers[0], ivec2(int(p.x) / 2, p.y / 2), 0);
            }else{
                p = mMatrices[1] * p + vec4(1, 1, 0, 0);
                p.x = p.x * (0.5*w);
                p.y = p.y * (0.5*h);
                frag_out = texelFetch(samplers[1], ivec2((int(p.x) -2) / 2, p.y / 2), 0);
            }
        } else {
          if(int(gl_FragCoord.x) % 2 == 0){
                p = mMatrices[0] * p + vec4(1, 1, 0, 0);
                p.x = p.x * (0.5*w);
                p.y = p.y * (0.5*h);
                frag_out = texelFetch(samplers[0], ivec2(int(p.x) / 2, p.y / 2), 1);
            }else{
                p = mMatrices[1] * p + vec4(1, 1, 0, 0);
                p.x = p.x * (0.5*w);
                p.y = p.y * (0.5*h);
                frag_out = texelFetch(samplers[1], ivec2(int(p.x) / 2, p.y / 2), 1);
            }  
        }

}
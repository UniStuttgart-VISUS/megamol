//#version 440
uniform sampler2DArray src_tex2D;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;
uniform int amortLevel;
uniform mat4 moveMatrices[49];
//layout (std430, binding = 7) buffer ssboMatrices{
//    mat4 mMatrices[];
//};

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
    int line = int(uv_coord.y * h) % amortLevel;
    int col = int(uv_coord.x * w) % amortLevel;
    vec4 p = vec4(2*uv_coord-vec2(1.0), 0.0, 1.0);
    int i = (amortLevel * line + col);

    float dist = 99999.9;

    //can be removed later
    vec4 ps = moveMatrices[i] * p;
    ps = (ps + vec4(1.0, 1.0, 0.0, 0.0)) /2;
    int kcol = int(ps.x * w) % amortLevel;
    int kline = int(ps.y * h) % amortLevel;
    int k = amortLevel * kline + kcol;

    if(k != i){
        int yOff = line - kline;
        int xOff = kcol - col;
        frag_out = texture(src_tex2D, vec3(0.5 * (moveMatrices[k] * (p + vec4(xOff / w, yOff / h, 0, 0))).xy + vec2(0.5), k));
        //frag_out = vec4(1,0,0,1);
        }else{
        frag_out = texture(src_tex2D, vec3(0.5 * (moveMatrices[i] * p).xy + vec2(0.5), i));
    }
    int xDist = 999;
    int yDist = 999;

    int eline;
    int ecol;
    for (int j = 0; j < amortLevel * amortLevel; j++){
        //probably needlessly complicated
        int lineOffset = line - int(floor(j / amortLevel));
        int colOffset = j % amortLevel;
        vec4 pj = moveMatrices[j] * (p + vec4(2 * colOffset / w, 2 * lineOffset /h, 0, 0));
        pj = (pj + vec4(1.0, 1.0, 0.0, 0.0)) /2;
        int jcol = int(pj.x * w) % amortLevel;
        int jline = int(pj.y * float(h))%amortLevel;

        if(abs(line - jline) < yDist || abs(col - jcol) < xDist){
             xDist = abs(col-jcol);
             yDist = abs(line - jline);
             ecol = jcol;
             eline = jline;
        }
    }   
    int e = amortLevel * eline + ecol;
    int lineOffset = line - int(floor(e / amortLevel));
    int colOffset = e % amortLevel;
    frag_out = texture(src_tex2D, vec3(0.5 * (moveMatrices[e] * (p + vec4(colOffset / w, lineOffset / h, 0, 0))).xy + vec2(0.5), e));

    frag_out = vec4(kcol / float(amortLevel), kline / float(amortLevel), 0 ,0);
    //frag_out = vec4(col / float(amortLevel), line / float(amortLevel),0,0);
    //frag_out = texture(src_tex2D, vec3(0.5 * (moveMatrices[e] * p).xy + vec2(0.5), e));
    //frag_out = p;
}

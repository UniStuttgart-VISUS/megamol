
uniform vec4 inConsts1;
attribute float colIdx;
uniform sampler1D colTab;
uniform mat4 mvp;
varying vec3 opos;

#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w

void main(void) {

    float cid = MAX_COLV - MIN_COLV;
    if (cid < 0.000001) {
        gl_FrontColor = gl_Color;
    } else {
        cid = (colIdx - MIN_COLV) / cid;
        cid = clamp(cid, 0.0, 1.0);

        cid *= (1.0 - 1.0 / COLTAB_SIZE);
        cid += 0.5 / COLTAB_SIZE;

        gl_FrontColor = texture1D(colTab, cid);
    }

    opos = gl_Vertex.xyz;
    gl_Position = mvp * gl_Vertex; // was: ftransform, probably just the normal mvp needed here
}

uniform mat4 modelview;
uniform mat4 proj;
uniform vec4 inConsts1;
uniform sampler1D colTab;

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;

in vec4 vertex;
in vec4 color;

out vec4 inColor;

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w

void main(void) {
    gl_Position = vertex;
    float rad = vertex.w;
    if (CONSTRAD > -0.5) {
      gl_Position.w = CONSTRAD;
      rad = CONSTRAD;
    }
    
    // clipping
    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        float od = dot(vertex.xyz, clipDat.xyz) - rad;
        if (od > clipDat.w) {
          gl_Position = vec4(1.0, 1.0, 1.0, 0.0);
        }
    }   
    
    if (COLTAB_SIZE > 0.0) {    
        float cid = MAX_COLV - MIN_COLV;    
        if (cid < 0.000001) {
            inColor = texture(colTab, 0.5 / COLTAB_SIZE);
        } else {
            cid = (color.r - MIN_COLV) / cid;
            cid = clamp(cid, 0.0, 1.0);
        
            cid *= (1.0 - 1.0 / COLTAB_SIZE);
            cid += 0.5 / COLTAB_SIZE;
        
            inColor = texture(colTab, cid);
        }
    } else {
        inColor = color;
    }
}